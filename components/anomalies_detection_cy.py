import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import warnings
import os


warnings.filterwarnings('ignore')


@dataclass
class AnomalyResult:
    category_group: Dict[str, Any]
    anomaly_type: str
    column: str
    date: Optional[str]
    expected_value: Optional[float]
    actual_value: Optional[float]
    severity: str
    details: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class DataAnomalyDetector:
    
    def __init__(self, date_column: str = 'date', category_columns: List[str] = None, volume_columns: List[str] = None):
        self.date_column = date_column
        self.category_columns = category_columns or []
        self.volume_columns = volume_columns or []
        self.missing_data_threshold = 1
        self.anomalies = []
    
    def load_data(self, file_path: str, sheet_name: str = None, **kwargs) -> pd.DataFrame:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs) if sheet_name else pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.date_column not in df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found")
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        return df.sort_values(self.date_column)
    
    def detect_missing_data(self, df: pd.DataFrame) -> List[AnomalyResult]:
        anomalies = []
        groups = [('All', df)] if not self.category_columns else list(df.groupby(self.category_columns))
        
        for group_key, group_data in groups:
            category_group = dict(zip(self.category_columns, group_key)) if isinstance(group_key, tuple) else {'group': 'All'}
            
            min_date, max_date = group_data[self.date_column].min(), group_data[self.date_column].max()
            date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            existing_dates = set(group_data[self.date_column].dt.date)
            
            current_missing_range = []
            for date in date_range:
                if date.date() not in existing_dates:
                    current_missing_range.append(date)
                else:
                    if len(current_missing_range) >= self.missing_data_threshold:
                        start_date = current_missing_range[0].strftime('%Y-%m-%d')
                        end_date = current_missing_range[-1].strftime('%Y-%m-%d')
                        severity = 'high' if len(current_missing_range) >= 7 else 'medium' if len(current_missing_range) >= 3 else 'low'
                        
                        anomalies.append(AnomalyResult(
                            category_group=category_group, anomaly_type='missing_data', column=self.date_column,
                            date=start_date, expected_value=None, actual_value=None, severity=severity,
                            details=f"Missing {len(current_missing_range)} consecutive days",
                            start_date=start_date, end_date=end_date
                        ))
                    current_missing_range = []

            if len(current_missing_range) >= self.missing_data_threshold:
                start_date = current_missing_range[0].strftime('%Y-%m-%d')
                end_date = current_missing_range[-1].strftime('%Y-%m-%d')
                severity = 'high' if len(current_missing_range) >= 7 else 'medium' if len(current_missing_range) >= 3 else 'low'
                
                anomalies.append(AnomalyResult(
                    category_group=category_group, anomaly_type='missing_data', column=self.date_column,
                    date=start_date, expected_value=None, actual_value=None, severity=severity,
                    details=f"Missing {len(current_missing_range)} consecutive days",
                    start_date=start_date, end_date=end_date
                ))
        
        return anomalies
    
    def detect_volume_anomalies(self, df: pd.DataFrame) -> List[AnomalyResult]:
        anomalies = []
        groups = [('All', df)] if not self.category_columns else list(df.groupby(self.category_columns))
        
        max_date = df[self.date_column].max()
        recent_week_end = max_date if max_date.weekday() == 5 else max_date - pd.Timedelta(days=(max_date.weekday() + 2) % 7)
        recent_week_start = recent_week_end - pd.Timedelta(days=6)
        
        mtd_end_date = recent_week_end
        current_month_start = mtd_end_date.replace(day=1)
        prev_year_month_start = current_month_start - pd.DateOffset(years=1)
        prev_year_mtd_end = mtd_end_date - pd.DateOffset(years=1)
        
        ytd_end_date = recent_week_end
        current_year_start = ytd_end_date.replace(month=1, day=1)
        prev_year_start = current_year_start - pd.DateOffset(years=1)
        prev_year_ytd_end = ytd_end_date - pd.DateOffset(years=1)
        
        for group_key, group_data in groups:
            category_group = dict(zip(self.category_columns, group_key)) if isinstance(group_key, tuple) else {'group': 'All'}
            
            for volume_col in self.volume_columns:
                if volume_col not in group_data.columns:
                    continue

                recent_week_data = group_data[(group_data[self.date_column] >= recent_week_start) & (group_data[self.date_column] <= recent_week_end)]
                if not recent_week_data.empty:
                    recent_week_total = recent_week_data[volume_col].sum()
                    prev_week_data = group_data[(group_data[self.date_column] >= recent_week_start - pd.Timedelta(days=7)) & 
                                               (group_data[self.date_column] <= recent_week_end - pd.Timedelta(days=7))]
                    
                    if not prev_week_data.empty:
                        prev_week_total = prev_week_data[volume_col].sum()
                        if prev_week_total > 0:
                            wow_change = (recent_week_total - prev_week_total) / prev_week_total
                            if abs(wow_change) > 0.3:
                                severity = 'high' if abs(wow_change) >= 1.0 else 'medium' if abs(wow_change) >= 0.5 else 'low'
                                anomalies.append(AnomalyResult(
                                    category_group=category_group, anomaly_type='abnormal_volume', column=volume_col,
                                    date=recent_week_end.strftime('%Y-%m-%d'), expected_value=prev_week_total,
                                    actual_value=recent_week_total, severity=severity,
                                    details=f"WoW: {wow_change:.1%} change (Recent: {recent_week_total:.0f}, Previous: {prev_week_total:.0f})"
                                ))
                
                current_mtd_data = group_data[(group_data[self.date_column] >= current_month_start) & (group_data[self.date_column] <= mtd_end_date)]
                prev_year_mtd_data = group_data[(group_data[self.date_column] >= prev_year_month_start) & (group_data[self.date_column] <= prev_year_mtd_end)]
                
                if not current_mtd_data.empty and not prev_year_mtd_data.empty:
                    current_mtd_total = current_mtd_data[volume_col].sum()
                    prev_year_mtd_total = prev_year_mtd_data[volume_col].sum()
                    
                    if prev_year_mtd_total > 0:
                        mtd_yoy_change = (current_mtd_total - prev_year_mtd_total) / prev_year_mtd_total
                        if abs(mtd_yoy_change) > 0.4:
                            severity = 'high' if abs(mtd_yoy_change) >= 1.0 else 'medium' if abs(mtd_yoy_change) >= 0.5 else 'low'
                            anomalies.append(AnomalyResult(
                                category_group=category_group, anomaly_type='abnormal_volume', column=volume_col,
                                date=mtd_end_date.strftime('%Y-%m-%d'), expected_value=prev_year_mtd_total,
                                actual_value=current_mtd_total, severity=severity,
                                details=f"MTD YoY: {mtd_yoy_change:.1%} change (Current: {current_mtd_total:.0f}, Last Year: {prev_year_mtd_total:.0f})"
                            ))
                
                current_ytd_data = group_data[(group_data[self.date_column] >= current_year_start) & (group_data[self.date_column] <= ytd_end_date)]
                prev_year_ytd_data = group_data[(group_data[self.date_column] >= prev_year_start) & (group_data[self.date_column] <= prev_year_ytd_end)]
                
                if not current_ytd_data.empty and not prev_year_ytd_data.empty:
                    current_ytd_total = current_ytd_data[volume_col].sum()
                    prev_year_ytd_total = prev_year_ytd_data[volume_col].sum()
                    
                    if prev_year_ytd_total > 0:
                        ytd_yoy_change = (current_ytd_total - prev_year_ytd_total) / prev_year_ytd_total
                        if abs(ytd_yoy_change) > 0.3:
                            severity = 'high' if abs(ytd_yoy_change) >= 1.0 else 'medium' if abs(ytd_yoy_change) >= 0.5 else 'low'
                            anomalies.append(AnomalyResult(
                                category_group=category_group, anomaly_type='abnormal_volume', column=volume_col,
                                date=ytd_end_date.strftime('%Y-%m-%d'), expected_value=prev_year_ytd_total,
                                actual_value=current_ytd_total, severity=severity,
                                details=f"YTD YoY: {ytd_yoy_change:.1%} change (Current: {current_ytd_total:.0f}, Last Year: {prev_year_ytd_total:.0f})"
                            ))
        
        return anomalies
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, List[AnomalyResult]]:
        print("Starting anomaly analysis...")
        df_processed = self.preprocess_data(df)
        missing_data_anomalies = self.detect_missing_data(df_processed)
        volume_anomalies = self.detect_volume_anomalies(df_processed)
        
        self.anomalies = missing_data_anomalies + volume_anomalies
        results = {'missing_data': missing_data_anomalies, 'abnormal_volume': volume_anomalies, 'all': self.anomalies}
        print(f"Analysis completed. Total anomalies found: {len(self.anomalies)}")
        return results
    
    def save_results(self, results: Dict[str, List[AnomalyResult]], report_path: str = None, excel_path: str = None):
        if excel_path:
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    summary_data = {
                        'Metric': ['Total Anomalies', 'Missing Data', 'Volume Anomalies', 'High Severity', 'Medium Severity', 'Low Severity'],
                        'Count': [len(results['all']), len(results['missing_data']), len(results['abnormal_volume']),
                                sum(1 for a in results['all'] if a.severity == 'high'),
                                sum(1 for a in results['all'] if a.severity == 'medium'),
                                sum(1 for a in results['all'] if a.severity == 'low')]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    if results['abnormal_volume']:
                        volume_data = []
                        for anomaly in results['abnormal_volume']:
                            record = {'Severity': anomaly.severity, 'Metrics': anomaly.column, 'Date': anomaly.date,
                                    'Previous': anomaly.expected_value, 'Current': anomaly.actual_value}
                            
                            if anomaly.category_group:
                                for cat_name, cat_value in anomaly.category_group.items():
                                    if cat_name != 'group':
                                        record[cat_name] = cat_value
                            
                            record['Details'] = anomaly.details
                            volume_data.append(record)
                        
                        pd.DataFrame(volume_data).to_excel(writer, sheet_name='Volume_Anomalies', index=False)
                    
                    if results['missing_data']:
                        missing_data = []
                        for anomaly in results['missing_data']:
                            record = {'Severity': anomaly.severity, 'Start_Date': anomaly.start_date, 
                                    'End_Date': anomaly.end_date, 'Details': anomaly.details}
                            
                            if anomaly.category_group:
                                for cat_name, cat_value in anomaly.category_group.items():
                                    if cat_name != 'group':
                                        record[cat_name] = cat_value
                            
                            missing_data.append(record)
                        
                        pd.DataFrame(missing_data).to_excel(writer, sheet_name='Missing_Data', index=False)
                
                print(f"Excel report saved: {excel_path}")
            except Exception as e:
                print(f"Error saving Excel file: {str(e)}")


def main():
    print("Data Anomaly Detection Platform")
    print("=" * 50)
    
    file_path = input("Enter the path to your data file (CSV or Excel): ").strip()
    if not os.path.exists(file_path):
        print("File not found.")
        return
    
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Available columns: {list(df.columns)}")
        
        date_column = input("Enter date column name: ").strip()
        if date_column not in df.columns:
            print("Date column not found.")
            return
        
        category_input = input("Enter category columns (comma-separated, or Enter to skip): ").strip()
        category_columns = [col.strip() for col in category_input.split(',')] if category_input else []
        
        volume_input = input("Enter volume columns (comma-separated): ").strip()
        if not volume_input:
            print("Volume columns required.")
            return
        volume_columns = [col.strip() for col in volume_input.split(',')]

        detector = DataAnomalyDetector(date_column=date_column, category_columns=category_columns, volume_columns=volume_columns)
        results = detector.analyze(df)
        
        print(f"Results: {len(results['all'])} total anomalies ({len(results['missing_data'])} missing data, {len(results['abnormal_volume'])} volume)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        input_dir = os.path.dirname(os.path.abspath(file_path))
        excel_file = os.path.join(input_dir, f"{base_name}_anomaly_details_{timestamp}.xlsx")
        
        detector.save_results(results, excel_path=excel_file)
        print(f"Analysis complete! Results saved to: {os.path.basename(excel_file)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


# if __name__ == "__main__":
#     main()
