#DATE BUCKETS FOR REPORTING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# For new marketing BRS digital wireline report format -------------CL-------------------
# 6 weeks, 7 days, MTD , YTD, in most week and their YOY ------------CL-------------------

def date_bucket_temp1 (report_date,current_date):

    import pandas as pd
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    from pandas.tseries.offsets import MonthEnd, QuarterEnd

    date_buckets = []
    for i in range(6):
            end_date = report_date - timedelta(days=i * 7)
            start_date = end_date - timedelta(days=6)
            date_buckets.append((pd.Timestamp(start_date), pd.Timestamp(end_date), f"Week {i + 1}"))

    # Calculate MTD (Month-to-Date)
    mtd_start_date = date_buckets[0][0].replace(day=1)
    mtd_end_date = current_date
    date_buckets.append((pd.Timestamp(mtd_start_date), pd.Timestamp(mtd_end_date), "MTD"))

    #date_obj = datetime.strptime(report_date, "%B %d, %Y")  # Convert to date
    day_value = report_date.day  
    if day_value < 7:
        # Find the index of the MTD entry in the list
        for idx, item in enumerate(date_buckets):
            if item[2] == "MTD":
                mtd_start_date = (item[1] - pd.DateOffset(months=1)).replace(day=1)  # Adjust start date
                mtd_end_date = mtd_start_date + MonthEnd(0)  # Adjust end date
                date_buckets[idx] = (mtd_start_date, mtd_end_date, "MTD")  # Replace tuple
                break  # Exit loop once modified

    
    # Calculate QTD (Quarter-to-Date)
    current_month = current_date.month
    quarter_start_month = ((current_month - 1) // 3) * 3 + 1  # Calculate the first month of the current quarter
    qtd_start_date = current_date.replace(month=quarter_start_month, day=1)
    qtd_end_date = current_date
    date_buckets.append((pd.Timestamp(qtd_start_date), pd.Timestamp(qtd_end_date), "QTD"))

    mth_value = report_date.month
    if day_value < 7 and mth_value in (4, 7, 10, 1):
        for idx, item in enumerate(date_buckets):
            if item[2] == "QTD":
                qtd_start_date = (item[1] - pd.DateOffset(months=3)).replace(day=1) # Adjust start date
                qtd_end_date = qtd_start_date + QuarterEnd(0)  # Adjust end date
                date_buckets[idx] = (qtd_start_date, qtd_end_date, "QTD")  # Replace tuple
                break  # Exit loop once modified

    # Calculate YTD (Year-to-Date)
    ytd_start_date = date_buckets[0][0].replace(month=1, day=1)
    ytd_end_date = current_date
    date_buckets.append((pd.Timestamp(ytd_start_date), pd.Timestamp(ytd_end_date), "YTD"))

    # transform to a dataframe for the date buckets
    date_buckets = pd.DataFrame(date_buckets, columns=['Start Date', 'End Date', 'Bucket Name'])
    date_buckets = date_buckets[['Bucket Name', 'Start Date', 'End Date']]

    # Create day 1 to 7 in Week 1
    week1_row = date_buckets[date_buckets["Bucket Name"] == "Week 1"].iloc[0]  # Extract Week 1 row
    week1_days = [
        week1_row["Start Date"] + timedelta(days=i) for i in range((week1_row["End Date"] - week1_row["Start Date"]).days + 1)
    ]

    day_rows = [{"Bucket Name": f"Day {i+1}", "Start Date": day, "End Date": day} for i, day in enumerate(week1_days)]

    # Append the new day rows to the original DataFrame
    date_buckets = pd.concat([date_buckets, pd.DataFrame(day_rows)], ignore_index=True)

    #-------------------------Create Last year YOY metrics accordingly --------------------------------------------
    LY_dates = []

    # Create LY weeks 1- 6 
    week_rows = date_buckets[date_buckets["Bucket Name"].str.contains("Week")]
    for i, row in week_rows.iterrows():
        LY_dates.append({
            "Bucket Name": f"LY{row['Bucket Name']}",
            "Start Date": row["Start Date"] - timedelta(weeks=52),
            "End Date": row["End Date"] - timedelta(weeks=52)
        })

    # Create LMTD (Last Month-to-Date)
    mtd_row = date_buckets[date_buckets["Bucket Name"] == "MTD"].iloc[0]
    LY_dates.append({
        "Bucket Name": "LYMTD",
        "Start Date": mtd_row["Start Date"] - relativedelta(months=12),
        "End Date": mtd_row["End Date"] - relativedelta(months=12)
    })

        # Create LQTD (Last Quarter-to-Date)
    qtd_row = date_buckets[date_buckets["Bucket Name"] == "QTD"].iloc[0]
    LY_dates.append({
        "Bucket Name": "LYQTD",
        "Start Date": qtd_row["Start Date"] - relativedelta(months=12),
        "End Date": qtd_row["End Date"] - relativedelta(months=12)
    })

    # Create LYTD (Last Year-to-Date)
    ytd_row = date_buckets[date_buckets["Bucket Name"] == "YTD"].iloc[0]
    LY_dates.append({
        "Bucket Name": "LYYTD",
        "Start Date": ytd_row["Start Date"] - relativedelta(months=12),
        "End Date": ytd_row["End Date"] - relativedelta(months=12)
     })
    ## Create LY days 1- 7 
    day_rows = date_buckets[date_buckets["Bucket Name"].str.contains("Day")]
    for i, row in day_rows.iterrows():
        LY_dates.append({
            "Bucket Name": f"LY{row['Bucket Name']}",
            "Start Date": row["Start Date"] - timedelta(weeks=52),
            "End Date": row["End Date"] - timedelta(weeks=52)
        })
   
    # Append the new rows to the DataFrame
    ly_date_buckets = pd.DataFrame(LY_dates)
    #adjust the end-date in LY week 1 given when the current date is not full week yet
    if current_date.date() < report_date :
        ly_date_buckets.loc[ly_date_buckets["Bucket Name"] == "LYWeek 1", "End Date"] = pd.to_datetime(current_date - timedelta(weeks=52))

    date_buckets = pd.concat([date_buckets, ly_date_buckets], ignore_index=True)

    return date_buckets 

#------for test ----------
# current_date= 2025-03-11
# report_date= 2025-03-15
# date_bucket_temp1 (report_date,current_date)

def agg_report(date_buckets, df):
    results = []

    # Guard: must have at least 2 columns (PERIOD_DT + value column)
    if df.shape[1] < 2:
        return []

    # Identify the column to aggregate (assumes 2nd column is value column)
    value_col = df.columns[1]

    for _, row in date_buckets.iterrows():
        bucket_name = row['Bucket Name']
        start_date = row['Start Date']
        end_date = row['End Date']

        # Filter by date
        filtered_data = df[(df['PERIOD_DT'] >= start_date) & (df['PERIOD_DT'] <= end_date)]

        # Sum the value column (if it's numeric)
        if pd.api.types.is_numeric_dtype(filtered_data[value_col]):
            value_sum = filtered_data[value_col].sum()
        else:
            value_sum = np.nan  # or 0, depending on what makes sense for you

        results.append({'Bucket Name': bucket_name, value_col: value_sum})

    return results

def linechart(df):
    if df.shape[1] < 2:
        st.warning("⚠️ Aggregated DataFrame must have at least two columns.")
        return

    try:
        # Extract week names and values
        week_names = df.iloc[:, 0].astype(str).tolist()
        week_values = df.iloc[:, 1].tolist()

        # Map week name to value
        week_to_value = dict(zip(week_names, week_values))

        # Separate weeks by current year and last year
        current_year_weeks = sorted(
            [w for w in week_names if w.startswith("Week")],
            key=lambda x: int(x.split(" ")[-1]),
            reverse=True
        )
        last_year_weeks = sorted(
            [w for w in week_names if w.startswith("LYWeek")],
            key=lambda x: int(x.replace("LYWeek", "")),
            reverse=True
        )

        # Create aligned data
        current_year_values = [week_to_value.get(w, np.nan) for w in current_year_weeks]
        last_year_values = [week_to_value.get(w, np.nan) for w in last_year_weeks]

        # Reverse order: Week 6 → Week 1
        labels = current_year_weeks

        # Plot chart
        fig, ax = plt.subplots()
        ax.plot(labels, current_year_values, marker='o', label='Current Year', color='steelblue')
        ax.plot(labels, last_year_values, marker='o', label='Last Year', color='orange')
        ax.set_title("📈 Weekly Comparison (Current vs Last Year)")
        ax.set_xlabel("Week")
        ax.set_ylabel("Metric")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"🚨 Chart generation failed: {e}")


