import streamlit as st

import pandas as pd
import numpy as np
from datetime import timedelta

import matplotlib.pyplot as plt



def plot_minute_level_data(program, df_features):
    minute_leveldata = df_features[df_features["program_name"] == program].copy()
    
    minute_leveldata = minute_leveldata[["program_name", "datetime_minute", 'predictions_euro', 'predictions_world_cup']].reset_index(drop=True).sort_values(by='datetime_minute')
    minute_leveldata["weighted_prediction"] = (minute_leveldata["predictions_euro"] * 0.6) +(minute_leveldata["predictions_world_cup"] * 0.4)
    minute_leveldata["min"] = minute_leveldata["weighted_prediction"]*0.8
    minute_leveldata["max"] = minute_leveldata["weighted_prediction"]*1.2

    # Plot actual and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(minute_leveldata['datetime_minute'], minute_leveldata['weighted_prediction'], label='Predicted', marker='o')
    plt.fill_between(minute_leveldata['datetime_minute'], minute_leveldata['min'], minute_leveldata['max'], color='gray', alpha=0.2, label='Prediction Range')

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


#Main Function


def main ():
    
    st.title('EURO 2024 Rating Predictions')

    df = pd.read_excel('best_training_predictions.v3.xlsx', sheet_name='euro_2024_predictions')


    selected_match_stage = st.sidebar.selectbox("Select Match Stage", df.match_stage_original.unique())

    if selected_match_stage == "GROUP STAGE":
        selected_group = st.sidebar.selectbox("Select Group", df.group_name.unique())
        filtered_match_stage = df[(df.match_stage_original == selected_match_stage) & (df.group_name == selected_group)]
    else:
        filtered_match_stage = df[df.match_stage_original == selected_match_stage]
        
    columns_to_display = ["program_name", "match_start_date",  "low", "weighted_prediction", "max"]
    #"channel_id_str", "group_name", "match_stage_8", "predictions_euro_2020", "predictions_world_cup",, 

    filtered_match_stage = filtered_match_stage[columns_to_display].rename(columns={"weighted_prediction": "prediction"})

    selection = dataframe_with_selections(filtered_match_stage)

    program_names = selection.program_name.unique()

    df_plot = pd.read_excel("euro_2024_predictions.xlsx")
    
    df_plot["match_start_date"] = pd.to_datetime(df_plot["match_start_date"])


    for program in program_names:
        plot_minute_level_data(program, df_plot)

if __name__ == "__main__":
    main()