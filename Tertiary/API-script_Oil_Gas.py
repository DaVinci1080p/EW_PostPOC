import csv
from datetime import datetime, timedelta

import requests


def fetch_and_save_data(api_key, product, csv_file_path, API_URL):
    # Construct the API URL with parameters
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    query = f"&frequency=daily&data[0]=value&facets[product][]={product}&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=14"
    url = f"{API_URL}{api_key}{query}"

    try:
        # Make the API request with a timeout of 5 seconds
        response = requests.get(url, timeout=5)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Load content into python format
            if "application/json" in response.headers.get("Content-Type", ""):
                json_data = response.json()

                # Check if the necessary data is present in the JSON structure
                if "response" in json_data and "data" in json_data["response"]:
                    data_list = json_data["response"]["data"]

                    # Read existing data from the CSV file
                    existing_data = []
                    try:
                        with open(
                            csv_file_path, mode="r", newline=""
                        ) as existing_csv_file:
                            csv_reader = csv.reader(existing_csv_file)
                            next(csv_reader)  # Skip header row
                            existing_data = list(csv_reader)
                    except FileNotFoundError:
                        pass

                    # Write the "period" and "value" aspects to the CSV file
                    with open(csv_file_path, mode="a", newline="") as csv_file:
                        csv_writer = csv.writer(csv_file)

                        # Write new data rows
                        for data_entry in data_list:
                            period = data_entry.get("period", [])
                            value = data_entry.get("value", [])

                            if period and value:
                                # Convert the string representation of the period to a common format
                                period = datetime.strptime(period, "%Y-%m-%d")
                                period = period.strftime("%d/%m/%Y")

                                # Check if the period is not in existing data before writing
                                if period not in (row[0] for row in existing_data):
                                    csv_writer.writerow([period, value])
                                    existing_data.append(
                                        [period, value]
                                    )  # Update the set of existing periods

                    print(f"Data has been updated and saved to {csv_file_path}")
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")

    except requests.Timeout:
        print("The request timed out. Please try again.")
    except requests.RequestException as e:
        print(f"An error occurred: {e}")


key = "vzsVfX0D3bV3xTQ0yGpnGLrsvFcqm0HYYa6LtLEB"
# URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key="
# resource = "EPCBRENT"
# csv_file_path = "C:/Users/RWend/OneDrive/Bureaublad/Brent_SpotPrice.csv"

URL = "https://api.eia.gov/v2/natural-gas/pri/fut/data/?api_key="
resource = "RNGWHHD"
path = "/Datasets/Gas_SpotPrice.csv"
fetch_and_save_data(key, resource, path, URL)
