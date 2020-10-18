# TODO: Refactor perform_action and parse_response
# so I don't need to have one for each type of action
import pandas as pd

def parse_response_hospitals(string_to_parse, country, data):
  return (string_to_parse % (
      country,
      data[0]['attributes']['name']
    )
  )

def parse_response_covid_metrics(string_to_parse, country, data):
  return (string_to_parse % (
      data['country'],
      pd.to_datetime(data['updated'], unit='ms').date(),
      data['todayCases'],
      data['cases']
    )
  )