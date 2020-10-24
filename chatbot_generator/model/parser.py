# TODO: Refactor perform_action and parse_response
# so I don't need to have one for each type of action
import pandas as pd

# Documentation for hospitals API: https://healthsites.io/api/docs/#
def parse_response_hospitals(string_to_parse, country, data, i):
  return (string_to_parse % (
      country,
      (i+1),
      len(data),
      data[i]['attributes']['name']
    )
  )

def parse_response_covid_metrics(string_to_parse, country, data, i):
  return (string_to_parse % (
      data['country'],
      pd.to_datetime(data['updated'], unit='ms').date(),
      data['todayCases'],
      data['cases']
    )
  )