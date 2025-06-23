import glob
import pandas as pd
import os

class DigitalSalesReport:
    def __init__(self, excel_folder="data"):
        
        # Find the latest matching file
        file_list = glob.glob(f"{excel_folder}/Digital Sales report*.xlsx") 
        self.excel_file = max(file_list, key=os.path.getctime) if file_list else None
        self.default_sheet = "Bell"  # Set your default fallback sheet

        # Load Excel file if found
        if self.excel_file:
            self.xls = pd.ExcelFile(self.excel_file) 
            print(f"Loaded file: {self.excel_file}")
        else:
            print("No matching file found.")

        # Keyword mapping with multiple keywords per sheet
        self.keyword_map = {
            "Bell": ["bell","bell sales", "bell online", "bell traffic", "bell trend", "bell performance"],
            "Virgin": ["virgin", "virgin online", "virgin traffic", "virgin trend","virgin performance"],
            "Bell Tech Error": ["error", "errors", "tech error", "bell errors"],
            "Bell BYOD ESIM": ["mobility", "byod", "byop", "esim"],
            "Bell Shop Traffic Soure": ["traffic source", "traffic mix"]
        }

        # Key metrics for each sheet
        self.sheet_metrics = {
            "Bell": ["Gross Sales", "Cusotmer new vs existing", "channel mix", "traffic mix", "shop traffic", "eshop vs App", "Mobility vs Residential", "regional"],
            "Virgin": ["Gross Sales", "Cusotmer new vs existing", "channel mix", "traffic mix", "shop traffic", "eshop vs App", "Mobility vs Residential", "Regional"],
            "Bell Tech Error": ["Web wireless new", "web wireless existing", "web wireline new",  "web wireline existing", "App wireless new", "App wireless existing ", "App wireline existing"],
            "Bell BYOD ESIM": ["sales", "adobe online journey", "app orders - prior carriers", "activation%"],
            "Bell Shop Traffic Soure": ["New vs Existing", "Wireless vs Wireline", "conversation%", "direct, organic search, paid search, display, campaign etc."]
        }

    def find_matching_sheet(self, user_input):
        """Find the corresponding sheet based on user input keywords."""
        for sheet, keywords in self.keyword_map.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                return sheet
        return None

    def get_metrics(self, user_input):
        """Retrieve key metrics for a given sheet name."""
        sheet_name = self.find_matching_sheet(user_input)

        return self.sheet_metrics.get(sheet_name, [])
    
    def load_dataframe(self, user_input):
        """Load the matched sheet into a DataFrame."""
        if not self.xls:
            return None  # No Excel file found

        sheet_name = self.find_matching_sheet(user_input)

        if sheet_name in self.xls.sheet_names:
            return pd.read_excel(self.xls, sheet_name=sheet_name)
        return None  # Return None if sheet isn't in the file