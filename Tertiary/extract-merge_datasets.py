import csv

import requests

api_key = "vzsVfX0D3bV3xTQ0yGpnGLrsvFcqm0HYYa6LtLEB"
API_URL = "https://api.eia.gov/v2/natural-gas/pri/fut/data/?api_key="
product = "RNGWHHD"
query = f"&frequency=daily&data[0]=value&facets[series][]={product}&start=1993-12-24&end=2005-11-24&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"
api_url = f"{API_URL}{api_key}{query}"

response = requests.get(api_url)
# Check if the request was successful (status code 200)
if response.status_code == 200:
    # If the content is in JSON format, you can also load it into a Python object
    if "application/json" in response.headers.get("Content-Type", ""):
        json_data = response.json()

        # Check if the necessary data is present in the JSON structure
        if "response" in json_data and "data" in json_data["response"]:
            data_list = json_data["response"]["data"]
            # Specify the CSV file path
            csv_file_path = "C:/Users/RWend/OneDrive/Bureaublad/test.csv"

            # Write the "period" and "value" aspects to the CSV file
            with open(csv_file_path, mode="w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)

                # Write header row
                csv_writer.writerow(["period", "value"])
                # Write data rows
                for data_entry in data_list:
                    period = data_entry.get("period", [])
                    value = data_entry.get("value", [])
                    # Write a row only if both "period" and "value" are present
                    if period and value:
                        csv_writer.writerow([period, value])

            print(f"Data has been saved to {csv_file_path}")
        else:
            print("Invalid JSON structure. Missing 'response' or 'data' key.")
    else:
        print("Response content is not in JSON format.")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
