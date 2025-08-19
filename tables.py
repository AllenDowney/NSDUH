import re
from bs4 import BeautifulSoup
import requests
import os
from urllib.parse import urlparse
import pandas as pd
import shelve

def download_nsduh_tables(filename="2024-nsduh-detailed-tables-sect1pe.htm"):
    """
    Download NSDUH detailed tables from SAMHSA website.
    
    Returns:
        str: Path to the downloaded file, or None if download failed
    """
    url = "https://www.samhsa.gov/data/sites/default/files/reports/rpt56484/NSDUHDetailedTabs2024/NSDUHDetailedTabs2024/2024-nsduh-detailed-tables-sect1pe.htm#tab1.23a"
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    filepath = os.path.join(data_dir, filename)
    
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
        
        # Persist mappings to shelve
        shelve_path = os.path.join(csv_dir, "tables_metadata")
        try:
            with shelve.open(shelve_path) as db:
                db["filename_to_title"] = filename_to_title
                db["filename_to_number"] = filename_to_number
            print(f"\nSaved filename-to-title and filename-to-number mappings to shelve: {shelve_path}")
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
    
    return df_clean


def add_diffs(df):
    """
    Add difference columns showing year-over-year changes for each age group.
    
    This function expects columns like:
    - 'Aged 12-17 (2023)', 'Aged 12-17 (2024)'
    - 'Aged 18+ (2023)', 'Aged 18+ (2024)'
    
    It adds new columns:
    - 'Aged 12-17 (2024-2023)': Change from 2023 to 2024
    - 'Aged 18+ (2024-2023)': Change from 2023 to 2024
    
    Args:
        df (pandas.DataFrame): DataFrame with year columns for different age groups
        
    Returns:
        pandas.DataFrame: DataFrame with additional difference columns
    """
    import pandas as pd
    import numpy as np
    
    # Create a copy to avoid modifying the original
    df_with_diffs = df.copy()
    
    # Define the age groups and years we're looking for
    age_groups = ['Aged 12-17', 'Aged 18+']
    years = [2023, 2024]
    
    # For each age group, calculate the difference
    for age_group in age_groups:
        # Find the columns for this age group
        col_2023 = f"{age_group} ({years[0]})"
        col_2024 = f"{age_group} ({years[1]})"
        
        # Check if both columns exist
        if col_2023 in df_with_diffs.columns and col_2024 in df_with_diffs.columns:
            # Create the difference column name
            diff_col = f"{age_group} (change)"
            
            # Calculate differences, handling non-numeric values
            try:
                # Convert to numeric, coercing errors to NaN
                values_2023 = pd.to_numeric(df_with_diffs[col_2023], errors='coerce')
                values_2024 = pd.to_numeric(df_with_diffs[col_2024], errors='coerce')
                
                # Calculate difference (2024 - 2023)
                diffs = values_2024 - values_2023
                
                # Add the difference column
                df_with_diffs[diff_col] = diffs
                
            except Exception as e:
                print(f"Warning: Could not calculate differences for {age_group}: {e}")
                # Add NaN column if calculation fails
                df_with_diffs[diff_col] = np.nan
    
    return df_with_diffs


if __name__ == "__main__":
    if True:
        # Download the tables when script is run directly
        filename = "2024-nsduh-detailed-tables-sect1pe.htm"
        html_filepath = download_nsduh_tables(filename)
        
        csv_files = extract_tables_to_csv(html_filepath)
        if csv_files:
            print(f"\nAll tables have been processed!")
            print(f"CSV files are located in: data/csv/")
            print(f"Total CSV files created: {len(csv_files)}")
        else:
            print("\nDownload failed. Please check the error messages above.")

        filename_to_title, filename_to_number = load_table_mappings()
        print(filename_to_title)
    
    
    

