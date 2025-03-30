import re
import json
import os

def extract_section(text, section_name):
    """
    Extracts the section content for a given section name.
    It looks for a header "### <section_name>" and returns all text until the next header or end-of-file.
    """
    pattern = re.compile(rf'###\s*{re.escape(section_name)}\n(.*?)(?=\n\n\s*###|\Z)', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None

def parse_markdown_table(section_text):
    """
    Given a section text, finds the first Markdown table and parses it into a list of dictionaries.
    Returns the parsed table data if found, else None.
    Assumes:
      - The first non-empty table row is the header.
      - The next row is the separator.
      - Subsequent rows are data rows.
    """
    lines = section_text.splitlines()
    table_start = None
    table_end = None

    # Find contiguous lines starting with "|"
    for i, line in enumerate(lines):
        if line.strip().startswith("|"):
            if table_start is None:
                table_start = i
            table_end = i
        elif table_start is not None:
            # End of table block when encountering a non-table line after starting.
            break

    if table_start is None:
        return None  # No table found

    table_lines = lines[table_start:table_end+1]
    # Remove empty lines
    table_lines = [line.strip() for line in table_lines if line.strip()]
    if len(table_lines) < 3:
        return None  # Not enough rows for a table

    # The first line is header; second is a separator; subsequent are data rows.
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    # Normalize header names (lowercase for keys)
    headers = [h.lower() for h in headers]

    rows = []
    for line in table_lines[2:]:
        if not line.startswith("|"):
            continue
        values = [v.strip() for v in line.strip("|").split("|")]
        if len(values) != len(headers):
            continue  # skip if row doesn't match header columns
        row_dict = {}
        for key, value in zip(headers, values):
            # Optionally, convert the "num" field to an integer.
            if key == "num":
                try:
                    row_dict[key] = int(value)
                except ValueError:
                    row_dict[key] = value
            else:
                row_dict[key] = value
        rows.append(row_dict)
    return rows

def process_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # We extract only the three specified sections.
    sections_to_extract = ["Description", "Action Space", "Observation Space"]
    extracted = {}
    txt_output = ""

    for section in sections_to_extract:
        content = extract_section(text, section)
        if content is None:
            print(f"Section '{section}' not found in {filename}.")
            continue

        # Append to text output (with heading)
        txt_output += f"### {section}\n{content}\n\n"

        # For Action Space and Observation Space, parse the markdown table.
        if section in ["Action Space", "Observation Space"]:
            table = parse_markdown_table(content)
            if table is not None:
                extracted[section] = table
            else:
                # If no table is found, store the raw text.
                extracted[section] = content
        else:
            # For Description, we just store the text.
            extracted[section] = content

    # Prepare output filenames.
    base_name = os.path.splitext(filename)[0]
    json_filename = f"{base_name}_obs.json"
    txt_filename = f"{base_name}_obs.txt"

    # Write JSON output.
    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(extracted, jf, indent=4)
    print(f"Saved JSON to {json_filename}")

    # Write text output.
    with open(txt_filename, "w", encoding="utf-8") as tf:
        tf.write(txt_output.strip())
    print(f"Saved text to {txt_filename}")

if __name__ == "__main__":
    # Example: List of Python files to process.
    for f in os.listdir('.'):
        if f.endswith('.py') and f != os.path.basename(__file__):
            process_file(f)
