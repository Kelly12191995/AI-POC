from anomalies_detection_cy import DataAnomalyDetector
from datetime import datetime
import os

# Change below to define your data column, category columns, and metrics 
detector = DataAnomalyDetector(
    date_column='PERIOD_DT',
    category_columns=['BRAND', 'CHANNEL', 'TAG'],
    volume_columns=['TOTAL_SESSIONS', 'TOTAL_REG_SESSIONS', 'COMPLETED', 'HAS_WARNING', 'HAS_ERROR', 'ERROR_ONLY', 'WARNING_ONLY']
)

# Change to your own data file path and please specify the sheet name if the input data file contains multiple worksheets 
data_file = r"C:\Users\hu6111254\OneDrive - Bell Canada\AD-HOC\Testing.xlsx"
df = detector.load_data(data_file, sheet_name='Raw Data')
results = detector.analyze(df)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
input_dir = os.path.dirname(os.path.abspath(data_file))
base_name = os.path.splitext(os.path.basename(data_file))[0]
report_file = os.path.join(input_dir, f"{base_name}_report_{timestamp}.txt")
excel_file = os.path.join(input_dir, f"{base_name}_details_{timestamp}.xlsx")
detector.save_results(results, report_file, excel_file)

