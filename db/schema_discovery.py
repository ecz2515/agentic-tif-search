import pandas as pd

def get_prompt_context(csv_path="data/tif_expenditures.csv") -> str:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"-- Failed to read CSV: {e}"

    columns_list = "\n".join([f"- `{col}`" for col in df.columns])
    districts = df['TIF District'].dropna().unique() if 'TIF District' in df.columns else []
    years = df['Report Year'].dropna().unique() if 'Report Year' in df.columns else []
    
    sample_query = ""
    if len(districts) > 0 and len(years) > 0:
        sample_query = f'''Example: SELECT SUM("Cost of Studies" + "Administrative Cost" + "Marketing Sites" + "Site Preparation Costs" + "Renovation, Rehab, Etc." + "Public Works" + "Removing Contaminants" + "Job Training" + "Financing Costs" + "Capital Costs" + "School Districts" + "Library Districts" + "Relocation Costs" + "In Lieu of Taxes" + "Job Training/Retraining" + "Interest Cost" + "New Housing" + "Day Care Services" + "Other") AS total_spent 
FROM expenditures 
WHERE "TIF District" = '{districts[0]}' AND "Report Year" = {years[0]}'''
    
    prompt = (
        "You are a SQL expert. You will be given a natural language query about TIF (Tax Increment Financing) expenditures.\n\n"
        "The data is stored in a table called 'expenditures' with the following columns:\n"
        f"{columns_list}\n\n"
        f"Sample districts: {', '.join(map(str, districts[:5]))}\n"
        f"Sample years: {', '.join(map(str, years[:5]))}\n\n"
        f"{sample_query}\n\n"
        "Convert the natural language query to SQL. Return ONLY the SQL query without any explanation.\n\n"
        "IMPORTANT: Column names must be quoted with double quotes since they contain spaces."
    )
    
    return prompt
