from bs4 import BeautifulSoup
import json
import base64
import zlib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def convert_base64_to_arr(data_string, dtype):
  binary_data = base64.b64decode(data_string)
  try:
    decompressed = zlib.decompress(binary_data)
  except zlib.error:
    decompressed = binary_data
  ndarr = np.frombuffer(decompressed, dtype=dtype)
  return ndarr


def find_keys_recursively(obj, target_key, target_value):
  results = []
  if isinstance(obj, dict):
    for key, value in obj.items():
      if key == target_key and value == target_value:
        results.append(obj)
      else:
        results.extend(find_keys_recursively(value, target_key, target_value))
  elif isinstance(obj, list):
    for item in obj:
      results.extend(find_keys_recursively(item, target_key, target_value))
  return results


def get_df_dict(html: str) -> dict[str,pd.DataFrame]:
  data_dict: dict[str,pd.DataFrame] = {}

  soup = BeautifulSoup(html, 'html.parser')
  json_scripts = soup.find_all("script", {"type": "application/json"})

  for tag in json_scripts:
    try:
      data = json.loads(tag.string)
      renderers = find_keys_recursively(data, 'name', 'GlyphRenderer')
      for r in renderers:
        name = r.get('attributes', {}).get('name', '')
        entries = r.get('attributes', {}).get('data_source', {}).get('attributes', {}).get('data', {}).get('entries', [])
        if len(entries) == 0:
          logger.error(f'entry not found for {name}')
        df = pd.DataFrame({})
        for entry in entries:
          axis_name = entry[0]
          type = entry[1].get('type')
          data = entry[1].get('array', {}).get('data')
          dtype = entry[1].get('dtype')
          if type == 'ndarray':
            ndarr = convert_base64_to_arr(data, dtype)
            df[axis_name] = ndarr
          else:
            logger.error(f'Unsupported type. {type}')
        data_dict[name] = df
    except json.JSONDecodeError as e:
      logger.error(f'JSON decode error: {e}')
  return data_dict


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('html_path', nargs=1, type=str)
  args = parser.parse_args()
  args.html_path = Path(args.html_path[0])
  logger.error(f'html_path = {args.html_path}')

  with open(args.html_path, 'r') as f:
    html = f.read()
  df_dict = get_df_dict(html)

  for key, df in df_dict.items():
    key = key.replace('/', '#')
    filename = f'{key}.csv'
    df.to_csv(filename, index=False)


if __name__ == '__main__':
  main()
