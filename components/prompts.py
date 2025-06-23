
def prompt(prompt_name):

    prompts = {
        "output_prompt": "Keep executive level tone, be concise with key data points and actionable insights in bullet points. Maximum 5 lines.",

        "Maintask": """You are an expert data and business intelligence analyst. 
        Use the data to analyze the week-over-week change in online sales for the most recent weeks, 
        and potential drivers for the up or down trends.""",

        "Sales_causes": """Online Sales consist of eshop sales and echat sales. Shop traffic positively drives online eshop sales. 
        Targeted Chats and chats assisted drive echat sales directly positively. 
        Sales trend can be explained in the areas of new vs existing customers, and region aspects: ON/QC/West/East. 
        Product mix has positive correlations with sales. 
        Ordering path drives sales positively. 
        Activation rate does not impact sales, sales happen before activation. 
        Online channel mix has positive correlations with sales.""",

        "LOO_understand": """week 6 is the week that in the past 6 weeks from now. week 1 is current week. 
        Analyse trend always from week 6 to week 1 as it is from old to recent. Show trend in an average % format for 6 weeks. and % for week1/week2-1 as wow
        LY means last year. Compare same week number with LY (last year) to get YOY numbers in %."""
    }

    return prompts.get(prompt_name, "Prompt not found")