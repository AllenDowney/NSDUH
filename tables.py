import re
import sys
from bs4 import BeautifulSoup
import requests
import os
from urllib.parse import urlparse
import pandas as pd
import shelve
import matplotlib.pyplot as plt


from utils import decorate

def download_nsduh_tables(filename="2024-nsduh-detailed-tables-sect1pe.htm"):
    """
    Download NSDUH detailed tables from SAMHSA website.
    
    Returns:
        str: Path to the downloaded file, or None if download failed
    """
    url_root = "https://www.samhsa.gov/data/sites/default/files/reports/rpt56484/NSDUHDetailedTabs2024/NSDUHDetailedTabs2024/"
    url = url_root + filename
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Extract filename from URL
    filepath = os.path.join(data_dir, filename)
    print(filename)
    print(url)

    # Check if file already exists
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        print(f"File size: {os.path.getsize(filepath):,} bytes")
        return filepath
    
    try:
        print(f"Downloading NSDUH detailed tables from: {url}")
        print(f"Saving to: {filepath}")
        
        # Download the file
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Successfully downloaded {len(response.text)} characters")
        print(f"File saved to: {filepath}")
        
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None
    except IOError as e:
        print(f"Error saving file: {e}")
        return None

def extract_tables_to_csv(html_filepath):
    """
    Extract all tables from the HTML file and save each as a CSV file.
    
    Args:
        html_filepath (str): Path to the HTML file
        
    Returns:
        list: List of CSV file paths that were created
    """
    try:
        print(f"Reading HTML file: {html_filepath}")
        
        # Read all tables from the HTML file
        tables = pd.read_html(html_filepath)
        
        print(f"Found {len(tables)} tables in the HTML file")
        
        # Extract table titles and numbers from HTML captions
        table_titles, table_numbers = extract_table_titles(html_filepath)
        
        csv_dir = "data"
        
        csv_files = []
        filename_to_title = {}
        filename_to_number = {}
        
        for i, table in enumerate(tables):
            # Generate CSV filename with three-digit numbering
            csv_filename = f"table_{table_numbers[i]}.csv"
            csv_filepath = os.path.join(csv_dir, csv_filename)
            
            try:
                # Save table to CSV
                table.to_csv(csv_filepath, index=False, encoding='utf-8')
                csv_files.append(csv_filepath)
                
                # Get table title and number (or use defaults if not found)
                title = table_titles.get(i, f"Table {i+1}")
                table_number = table_numbers.get(i, f"Table_{i+1}")
                
                # Track mappings for shelve
                filename_to_title[csv_filename] = title
                filename_to_number[csv_filename] = table_number
                
                print(f"  Table {i+1:03d} ({table_number}): {table.shape[0]} rows × {table.shape[1]} columns → {csv_filename}")
                print(f"    Title: {title}")
                
            except Exception as e:
                print(f"  Error saving table {i+1}: {e}")
                continue
        
        # Persist mappings to shelve (append mode)
        shelve_path = os.path.join("data", "tables_metadata")
        try:
            # Load existing mappings if they exist, otherwise start from scratch
            try:
                existing_filename_to_title, existing_filename_to_number = load_table_mappings(shelve_path)
            except Exception as e:
                print(f"Warning: Could not load existing mappings: {e}")
                existing_filename_to_title = {}
                existing_filename_to_number = {}
            
            # Merge existing and new mappings
            merged_filename_to_title = {**existing_filename_to_title, **filename_to_title}
            merged_filename_to_number = {**existing_filename_to_number, **filename_to_number}
            
            # Save merged mappings back to shelve
            with shelve.open(shelve_path) as db:
                db["filename_to_title"] = merged_filename_to_title
                db["filename_to_number"] = merged_filename_to_number
            
            print(f"\nSaved filename-to-title and filename-to-number mappings to shelve: {shelve_path}")
            print(f"  Existing mappings: {len(existing_filename_to_title)}")
            print(f"  New mappings: {len(filename_to_title)}")
            print(f"  Total mappings: {len(merged_filename_to_title)}")
            
        except Exception as e:
            print(f"\nWarning: Failed to save shelve metadata ({e})")

        print(f"\nSuccessfully extracted {len(csv_files)} tables to CSV files")
        return csv_files

    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return []

def extract_table_titles(html_filepath):
    """
    Extract table titles and numbers from HTML caption tags using BeautifulSoup.
    
    Args:
        html_filepath (str): Path to the HTML file
        
    Returns:
        tuple: (table_titles_dict, table_numbers_dict)
    """
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Find all caption tags
        caption_tags = soup.find_all('caption')
        
        # Create mappings of table index to title and table number
        table_titles = {}
        table_numbers = {}
        
        for i, caption in enumerate(caption_tags):
            # Get the text content and clean it up
            title = caption.get_text(strip=True)
            # Clean up HTML entities
            title = title.replace('–', '-').replace('\xa0', ' ').strip()
            table_titles[i] = title
            
            # Extract table number (e.g., "1.23A" from "Table 1.23A - ...")
            table_number_match = re.search(r'Table\s+(\d+\.\d+[A-Z]?)', title)
            if table_number_match:
                table_numbers[i] = table_number_match.group(1)
                print(f"Table {i}: {table_numbers[i]}")
            else:
                print("no match")
                table_numbers[i] = f"Table_{i+1}"
        
        return table_titles, table_numbers
        
    except Exception as e:
        print(f"Warning: Could not extract table titles: {e}")
        return {}, {}


def load_table_mappings(shelve_path=os.path.join("data", "tables_metadata")):
    """
    Load filename-to-title and filename-to-number mappings from shelve.

    Args:
        shelve_path (str): Path (without extension) to the shelve database.

    Returns:
        tuple[dict, dict]: (filename_to_title, filename_to_number)
    """
    try:
        with shelve.open(shelve_path) as db:
            filename_to_title = db.get("filename_to_title", {})
            filename_to_number = db.get("filename_to_number", {})
            return filename_to_title, filename_to_number
    except Exception as e:
        print(f"Error loading mappings from shelve '{shelve_path}': {e}")
        return {}, {}

    
import re

def parse_table_title(title):
    """
    Parse NSDUH table title into structured components.
    
    Args:
        title (str): Full table title like "Table 1.23B - Illicit Drug Use in Lifetime: Among People Aged 12 or Older; by Age Group and Demographic Characteristics, Percentages, 2023 and 2024"
    
    Returns:
        dict: Parsed components
    """
    # Extract table number (e.g., "1.23B")
    table_number_match = re.search(r'Table\s+(\d+\.\d+[A-Z]?)', title)
    table_number = table_number_match.group(1) if table_number_match else None
    
    # Split on the first dash to separate table number from description
    if ' - ' in title:
        parts = title.split(' - ', 1)
        table_info = parts[0]  # "Table 1.23B"
        description = parts[1]  # Everything after the dash
    else:
        description = title
    
    # Extract years if present
    years_match = re.search(r'(\d{4})\s+and\s+(\d{4})', description)
    years = (years_match.group(1), years_match.group(2)) if years_match else None
    
    # Extract measurement type (Numbers vs Percentages)
    measurement_type = None
    if 'Numbers in Thousands' in description:
        measurement_type = 'Numbers in Thousands'
    elif 'Percentages' in description:
        measurement_type = 'Percentages'
    
    # Extract time period
    time_period = None
    time_keywords = ['Lifetime', 'Past Year', 'Past Month']
    for keyword in time_keywords:
        if keyword in description:
            time_period = keyword
            break
    
    # Extract population
    population_match = re.search(r'Among People Aged ([^;]+)', description)
    population = population_match.group(1) if population_match else None
    
    # Extract breakdown variables - look for "by" but stop at the first semicolon
    breakdown = None
    if ' by ' in description:
        # Find the "by" section and extract until the first semicolon
        by_index = description.find(' by ')
        if by_index != -1:
            after_by = description[by_index + 4:]  # Skip " by "
            # Look for the first semicolon after "by"
            semicolon_index = after_by.find(';')
            if semicolon_index != -1:
                breakdown = after_by[:semicolon_index].strip()
            else:
                # If no semicolon, take everything after "by" but before measurement type
                breakdown = after_by.strip()
                # Remove measurement type if it's at the end
                if measurement_type and breakdown.endswith(measurement_type):
                    breakdown = breakdown[:-len(measurement_type)].strip()
                # Remove years if they're at the end
                if years and breakdown.endswith(f"{years[0]} and {years[1]}"):
                    breakdown = breakdown[:-len(f"{years[0]} and {years[1]}")].strip()
    
    return {
        'table_number': table_number,
        'full_description': description,
        'years': years,
        'measurement_type': measurement_type,
        'time_period': time_period,
        'population': population,
        'breakdown_variables': breakdown,
        'original_title': title
    }


def clean_table(df):
    """Replace strings with NaN and remove footnote letters.
    
    Args:
        df (pandas.DataFrame): DataFrame with potential footnote letters
        
    Returns:
        pandas.DataFrame: DataFrame with footnote letters removed
    """
    import pandas as pd
    import numpy as np
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Define the pattern for footnote letters (a-z, A-Z)
    footnote_pattern = r'[a-zA-Z]$'
    
    def clean_value(value):
        """Clean a single value by removing footnote letters."""
        if pd.isna(value) or not isinstance(value, str):
            return value
        
        # Remove footnote letters from the end
        import re
        cleaned = re.sub(footnote_pattern, '', value.strip())
        
        # Check if the cleaned value contains any numbers
        if re.search(r'\d', cleaned):
            return cleaned
        else:
            # If no numbers, replace with NaN
            return np.nan
    
    # Apply the cleaning function to all string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':  # String/object columns
            df_clean[col] = df_clean[col].apply(clean_value)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean


def add_diffs(df, years, group):
    """
    Add difference columns showing year-over-year changes and male-female differences.
    
    This function expects columns like:
    - 'Aged 12-17 (2023)', 'Aged 12-17 (2024)'
    - 'Aged 18+ (2023)', 'Aged 18+ (2024)'
    
    It adds new columns:
    - 'Aged 12-17 (change)': Change from 2023 to 2024
    - 'Aged 18+ (change)': Change from 2023 to 2024
    - 'Aged 12-17 (diff)': Male - Female difference for 12-17 age group
    - 'Aged 18+ (diff)': Male - Female difference for 18+ age group
    
    Args:
        df (pandas.DataFrame): DataFrame with year columns for different age groups
        
    Returns:
        pandas.DataFrame: DataFrame with additional difference columns
    """
    df_group = pd.DataFrame()

    # Create a copy to avoid modifying the original
    year0, year1 = years    
    col_year0 = f"{group} ({year0})"
    col_year1 = f"{group} ({year1})"

    df_group[year0] = df[col_year0]
    df_group[year1] = df[col_year1]
    changes = df_group[year1] - df_group[year0]
    diff = df_group[year1].diff()

    # Add the change column
    df_group["Change"] = changes
    df_group["Diff"] = diff
    
    return df_group


def make_table(df, rows=None, groups=None, years=None):
    """Make a table with the following columns:
    - 'Aged 12-17 (2023)', 'Aged 12-17 (2024)', 'Aged 12-17 (change)', 'Aged 12-17 (diff)'
    - 'Aged 18+ (2023)', 'Aged 18+ (2024)', 'Aged 18+ (change)', 'Aged 18+ (diff)'
    """
    if rows is None:
        rows = ['Male', 'Female']
    if groups is None:
        groups = ['Aged 12-17', 'Aged 18+']
    if years is None:
        years = [2023, 2024]

    table_dict = {}
    for group in groups:
        if years:
            df_group = add_diffs(df.loc[rows], years, group)
        else:
            df_group = df.loc[rows, group]
        table_dict[group] = df_group

    return pd.concat(table_dict, axis=1)

def clean_labels(df, label_prefix="Misuse of "):
    """
    Remove label_prefix text from the first level of a multiindex without changing order.
    
    This function is useful for cleaning NSDUH table labels where label_prefix 
    appears in the first level of the multiindex, such as:
    - "Misuse of Prescription Opioids" -> "Prescription Opioids"
    - "Misuse of Prescription Stimulants" -> "Prescription Stimulants"
    - "Any Fentanyl Use" -> "Fentanyl Use"
    
    Args:
        df (pandas.DataFrame): DataFrame with a multiindex
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned multiindex labels
    """
    # Check if the dataframe has a multiindex
    if not isinstance(df.index, pd.MultiIndex):
        print("Warning: DataFrame does not have a multiindex")
        return df
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Get the first level of the multiindex
    first_level = df_clean.index.get_level_values(0)
    
    # Clean the labels by removing label_prefix
    cleaned_labels = []
    for label in first_level:
        if isinstance(label, str) and label.startswith(label_prefix):
            cleaned_labels.append(label[len(label_prefix):])
        else:
            cleaned_labels.append(label)
    
    # Create new multiindex with cleaned first level
    new_index = pd.MultiIndex.from_arrays([
        cleaned_labels,
        *[df_clean.index.get_level_values(i) for i in range(1, df_clean.index.nlevels)]
    ], names=df_clean.index.names)
    
    # Assign the new index to the dataframe
    df_clean.index = new_index
    
    return df_clean

filename_to_title, filename_to_number = load_table_mappings()


def prepare_table(table, section=1, suffix='B', 
                   rows=None, groups=None, years=None):
    number = f'{section}.{table}{suffix}'
    filename = f'table_{number}.csv'
    title = filename_to_title[filename]
    table_info = parse_table_title(title)
    desc = table_info['full_description']
    key = desc.split(" in ")[0]
    
    df = pd.read_csv(f"data/{filename}", index_col=0)
    df = clean_table(df)
    return make_table(df, rows, groups, years)

def compile_tables(tables, section=1, suffix='B', 
                   rows=None, groups=None, years=None):
    table_dict = {}

    for table in tables:
        table_dict[table] = prepare_table(table, section, suffix, rows, groups, years)
        print(table)
        
    table = pd.concat(table_dict)
    return table

def plot_percentages(table, **options):
    table[2024].unstack(sort=False).plot(kind='barh')
    plt.gca().invert_yaxis()
    decorate(xlabel='Percent', **options)


def plot_changes(table, **options):
    table['Change'].unstack(sort=False).plot(kind='barh')
    plt.gca().invert_yaxis()
    decorate(xlabel='Change in percentage points', loc='lower right', **options)


def process_section(section):
    """
    Process a section of the NSDUH detailed tables.
    
    Args:
        section (str): The section of the NSDUH detailed tables to process.
        
    """
    filename = f"2024-nsduh-detailed-tables-{section}.htm"
    html_filepath = download_nsduh_tables(filename)
    print(html_filepath)
    
    csv_files = extract_tables_to_csv(html_filepath)
    if csv_files:
        print(f"\nAll tables have been processed!")
        print(f"CSV files are located in: data/csv/")
        print(f"Total CSV files created: {len(csv_files)}")
    else:
        print("\nDownload failed. Please check the error messages above.")

    filename_to_title, filename_to_number = load_table_mappings()
    print(filename_to_title)


if __name__ == "__main__":
    section = 'sect7pe'
    process_section(section)
    
    

