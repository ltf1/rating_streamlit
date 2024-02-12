import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn import metrics as met
import matplotlib.pyplot as plt
import shap
import xgboost as xgb


# functions

@st.cache_data
def load_data():
    data = pd.read_excel("euro_2024_fixture.xlsx")
    return data

# Function to create new rows
def expand_rows(row):
    start_time = row['match_start_date'] - timedelta(minutes=15)
    end_time = row['match_start_date'] + timedelta(minutes=120)
    minutes_range = pd.date_range(start=start_time, end=end_time, freq='1min')
    return pd.DataFrame({'program_name': row['program_name'], 'datetime_minute': minutes_range})

def PopularityScore(team1, team2, score_total):
    if ((team1 == 5) | (team2 == 5)) & ((team1 < 2.5) | (team2 < 2.5)):
        return (score_total +1) * 1.3
    
    elif ((team1 == 5) | (team2 == 5)) & ((team1 >= 2.5) | (team2 >= 2.5)):
        return (score_total) * 1.3
    
    elif (team1 >= 3) & (team2 >= 3):
        return score_total * 1.1
    
    elif (team1 > 3) | (team2 > 3):
        return score_total * 1.2
    
    elif (team1 == 3) | (team2 == 3):
        return score_total * 1.1

    else:
        return score_total
    
def CreateBucketKnockout2(diff_from_start_time, match_status):

    if diff_from_start_time< 0:
        return 'pre_match'
    elif diff_from_start_time <= 15:
        return 'first_quarter'
    elif diff_from_start_time <= 30:
        return 'second_quarter'
    elif diff_from_start_time <= 47:
        return 'third_quarter'
    elif diff_from_start_time in (48, 49, 59, 60,61,62):
        return 'break_time'
    elif 63 <= diff_from_start_time <= 77:
        return 'fifth_quarter'
    elif 78 <= diff_from_start_time <= 92:
        return 'sixth_quarter'
    elif 93 <= diff_from_start_time <= 115:
        return 'seventh_quarter'

    
    elif diff_from_start_time > 115 and match_status == 'normal':
        return "post_match"
    elif diff_from_start_time > 115 and diff_from_start_time <= 160 and match_status == 'extra_time':
        return "extra_time"
    elif diff_from_start_time > 115 and diff_from_start_time <= 160 and match_status == 'penalty':
        return "extra_time"
    
    elif diff_from_start_time > 160 and diff_from_start_time <= 175 and match_status == 'penalty':
        return "penalty"
    
    elif diff_from_start_time > 160 and match_status == 'extra_time':
        return "post_match"
    elif diff_from_start_time > 175 and match_status == 'penalty':
        return "post_match"
    else:
        return "control"
    

def MatchPeriod(hour):

    if hour <= 16:

        return 'OPT'
    
    elif hour <=19:

        return 'PT-1'
    
    elif hour <=22:

        return 'PT-2'
    

def CalculateMeanShapValues(xgb_model, df, select_columns):
    # Adding Shap Values
    shap.initjs()

    # Assuming xgb_model["preprocessor"].transform(df) returns the transformed data
    transformed_data = xgb_model["preprocessor"].transform(df[select_columns])

    # Get the column names of the transformed DataFrame
    column_names = xgb_model["preprocessor"].get_feature_names_out(input_features=df[select_columns].columns)

    # Create a new DataFrame with the transformed data and column names
    transformed_df = pd.DataFrame(transformed_data, columns=column_names)
    
    # Get XGBoost regressor from the Pipeline
    xgb_regressor = xgb_model['regressor']

    explainer = shap.Explainer(xgb_regressor)

    shap_values = explainer(transformed_df)

    # Extract SHAP values for each feature
    values = shap_values.values

    base_values = shap_values.base_values
    df_values = pd.DataFrame(values, columns=transformed_df.columns)

    df_values.columns = [f"shap_{col}" for col in df_values.columns]
    df_values["base_values"] = base_values

    df = pd.concat([df, df_values], axis=1)

    # Aggregated Shap Values
    shap_columns = df.filter(regex='^shap_', axis=1).columns.tolist()

    mean_shap_values = df[["program_name"] + shap_columns].groupby('program_name').mean()

    mean_shap_values.columns = [col.replace('shap_remainder__', '').replace('shap_cat__', '') for col in mean_shap_values.columns]
    
    return mean_shap_values



def plot_minute_level_data(program, df_features):
    minute_leveldata = df_features[df_features["program_name"] == program].copy()
    minute_leveldata = minute_leveldata[["program_name", "datetime_minute", "weighted_predictions", "lower_bound", "upper_bound"]].reset_index(drop=True).sort_values(by='datetime_minute')

    # Plot actual and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(minute_leveldata['datetime_minute'], minute_leveldata['weighted_predictions'], label='Predicted', marker='o')
    plt.fill_between(minute_leveldata['datetime_minute'], minute_leveldata['lower_bound'], minute_leveldata['upper_bound'], color='gray', alpha=0.2, label='Prediction Range')

    # Set labels and title
    plt.xlabel('Date Time Minute')
    plt.ylabel('Rating Values')
    plt.title(f'{program}')

    # Add legend
    plt.legend()

    # Show the graph for each program
    st.pyplot(plt)

def dataframe_with_selections(df: pd.DataFrame, init_value: bool = False, width: int = 1200) -> pd.DataFrame:
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", init_value)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        width=width,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

def plot_top_ten_abs_shap_values(program_name, mean_shap_values):
    # Select mean SHAP values for the specified program
    program_mean_shap_values = mean_shap_values.loc[program_name]
    
    # Sort mean SHAP values in descending order of absolute values and select top ten
    top_ten_abs_shap_values = program_mean_shap_values.abs().nlargest(10)
    
    # Get the real values for the top ten features
    top_ten_features = top_ten_abs_shap_values.index
    top_ten_real_values = program_mean_shap_values.loc[top_ten_features]
    
    # Plot top ten mean SHAP values horizontally with real values as labels
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    top_ten_real_values.sort_values().plot(kind='barh', colormap='viridis', alpha=0.7)
    plt.xlabel('Mean SHAP Value based on World Cup')
    plt.ylabel('Feature')
    plt.title(f'Top Ten Mean SHAP Values (Absolute) for Program: {program_name}')
    
    # Add real values as labels on the plot
    for index, value in enumerate(top_ten_real_values.sort_values()):
        plt.text(value, index, f'{value:.2f}', ha='left', va='center')
    
    st.pyplot(plt)
    
    


with open ('euro2024_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)




#Main Function
def main():
    st.title('EURO 2024 Rating Predictions')

    df = load_data()
    
    df["program_name"] = df["first_team"] + " - " + df["second_team"] + " Karsılasması"
    df = df[df.program_name.isnull() == False].reset_index(drop=True).copy()

    # Apply function to each row and concatenate the results
    new_rows = pd.concat([expand_rows(row) for _, row in df.iterrows()], ignore_index=True)

    df = pd.merge(new_rows, df, on="program_name", how="left")
    
    team_features = pd.read_excel('euro_24_final.xlsx', sheet_name='popularity_score')
    team_features.rename(columns={'second_team': 'team'},inplace=True)

    df_merge_first_team = pd.merge(df,team_features,left_on ='first_team',right_on ='team')
    df_merge_first_team.rename(columns= {'popularity_10_base' : 'popularity_10_base_first_team','popularity_5_base' : 'popularity_5_base_first_team',
                                         'fifa ranking' : 'fifa_ranking_first_team', 'team value': 'team_value_first_team', 
                                         "categorical_popularity" : "categorical_popularity_first_team"},inplace=True)

    df_merge_second_team = pd.merge(df,team_features,left_on ='second_team',right_on ='team')
    
    df_merge_second_team.rename(columns= {'popularity_10_base' : 'popularity_10_base_second_team', 'popularity_5_base' : 'popularity_5_base_second_team',
                                          'fifa ranking' : 'fifa_ranking_second_team', 'team value': 'team_value_second_team', 
                                          "categorical_popularity" : "categorical_popularity_second_team"},inplace=True)
    
    df_merge_second_team = df_merge_second_team[['program_name','datetime_minute','fifa_ranking_second_team','team_value_second_team',
                                                 'popularity_10_base_second_team', 'popularity_5_base_second_team',"categorical_popularity_second_team"]]


    df_features =  df_merge_first_team.merge(df_merge_second_team, how='inner', on=['program_name', 'datetime_minute'])
    
    
    
    # Calculate overall scores for each row
    df_features['score_total'] = (df_features['popularity_5_base_first_team'] + df_features['popularity_5_base_second_team'])
        
    df_features['score_3'] = df_features.apply(lambda x: PopularityScore(x['popularity_5_base_first_team'], x['popularity_5_base_second_team'], x['score_total']), axis=1)
    
    max_possible_score = (5 + 4.5) * 1.3 
    df_features['normalized_score_3'] = df_features['score_3'] / max_possible_score


    popular_team_list_2 = ["ARJANTIN", "PORTEKIZ",  "BREZILYA", "FRANSA"]
    df_features["one_popular_2"] = np.where(((df_features["first_team"].isin(popular_team_list_2)) | (df_features["second_team"].isin(popular_team_list_2)))  ,1,0)

    df_features["diff_from_start_time"] =(df_features.datetime_minute - df_features.match_start_date).dt.total_seconds()/60
    
    df_features["match_status"] = "normal"

    df_features['match_bucket_knockout_2'] = df_features.apply(lambda row: CreateBucketKnockout2(row['diff_from_start_time'],row['match_status']),axis=1)
    
    df_features = df_features[(df_features.diff_from_start_time <= 49) | 
                                                    (df_features.diff_from_start_time >= 59)].reset_index(drop=True)
    
    df_features["hour"] = df_features.datetime_minute.dt.hour
    df_features['hour_sin'] = np.sin(df_features.hour*(2.*np.pi/24))
    df_features['hour_cos'] = np.cos(df_features.hour*(2.*np.pi/24))

    df_features['match_start_day'] = df_features.match_start_date.dt.day_of_week

    df_features["new_week_cat"] = np.where(df_features.match_start_day == 5, "saturday", np.where(df_features.match_start_day == 6, "sunday", "weekday"))
    
    df_features["match_start_hour"] = df_features.match_start_date.dt.hour

    df_features['match_period'] = df_features.match_start_hour.apply(MatchPeriod)
    
    df_features["channel_id_str"] = df_features.channel_id.astype(str)
    
    
    # Making Predictions
    
    select_columns = ['hour_sin', 'hour_cos', 'new_week_cat',
       'tournament', 'acılıs_mac',"score_total","one_popular_2",
        'concurrent_match_4', 'match_stage_8',
        'match_period', 'channel_id_str',
       'diff_from_start_time',
        'match_bucket_knockout_2']
    
    
    X_predict_euro_2020 = df_features[select_columns]
    
    predictions_euro_2020 = xgb_model.predict(X_predict_euro_2020)
    
    df_features['predictions_euro_2020'] = predictions_euro_2020
    
    df_features["tournament"] = np.where(df_features.tournament == "AVRUPA SAMPIYONASI", "DUNYA KUPASI", df_features.tournament)
    
    
    X_predict_world_cup = df_features[select_columns]
    
    predictions_world_cup = xgb_model.predict(X_predict_world_cup)
    
    df_features['predictions_world_cup'] = predictions_world_cup
    
    df_features["weighted_predictions"] = (df_features.predictions_euro_2020*0.6 + df_features.predictions_world_cup*0.4).round(2)
    
    df_features["lower_bound"] = (df_features["weighted_predictions"] * 0.8).round(2)
    
    df_features["upper_bound"] = (df_features["weighted_predictions"] * 1.2).round(2)
    
    # SHAP Values
    
    mean_shap_values = CalculateMeanShapValues(xgb_model, df_features, select_columns)
    
    # Aggregated Results

    df_results = df_features.groupby(['program_name', "match_start_date", "channel_id_str", "group_name",
                                 "match_stage_8" ])[["predictions_euro_2020", "predictions_world_cup"]].mean().sort_values(by='match_start_date', ascending=True).reset_index()
    
    
    df_results["weighted_predictions"] = (df_results.predictions_euro_2020*0.6 + df_results.predictions_world_cup*0.4).round(2)
    
    df_results["lower_bound"] = (df_results["weighted_predictions"] * 0.8).round(2)
    
    df_results["upper_bound"] = (df_results["weighted_predictions"] * 1.2).round(2)
    

    
    
    selected_match_stage = st.sidebar.selectbox("Select Match Stage", df_results.match_stage_8.unique())
    
    if selected_match_stage == "GROUP STAGE":
        selected_group = st.sidebar.selectbox("Select Group", df_results.group_name.unique())
        filtered_match_stage = df_results[(df_results.match_stage_8 == selected_match_stage) & (df_results.group_name == selected_group)]
    else:
        filtered_match_stage = df_results[df_results.match_stage_8 == selected_match_stage]
        
    columns_to_display = ["program_name", "match_start_date",  "lower_bound", "weighted_predictions", "upper_bound"]
    #"channel_id_str", "group_name", "match_stage_8", "predictions_euro_2020", "predictions_world_cup",, 
    
    filtered_match_stage = filtered_match_stage[columns_to_display].rename(columns={"weighted_predictions": "prediction", "lower_bound": "min", "upper_bound": "max"})

    selection = dataframe_with_selections(filtered_match_stage)
    
    program_names = selection.program_name.unique()
    
    col1, col2 = st.columns(2)
    
    # Iterate over each program and display plots in two columns
    for program in program_names:
        # Plot minute-level predictions in the first column
        with col1:
            plot_minute_level_data(program, df_features)

        # Plot SHAP values in the second column
        with col2:
            plot_top_ten_abs_shap_values(program, mean_shap_values)

    
if __name__ == "__main__":
    main()


    