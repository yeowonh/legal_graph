
# PDF to DOCX to Markdown Converter

This script converts a PDF file into a DOCX file and then converts the DOCX file to a Markdown (MD) format. The script also removes the first three lines of the DOCX file after conversion.

## Features

- Convert PDF to DOCX using `pdf2docx`
- Remove the first three lines from the DOCX file
- Convert DOCX to Markdown using `Pandoc`
- Save output files in a user-specified directory

## Prerequisites

To run this script, you need to have the following software installed:

### 1. Python (3.6 or above)
Ensure you have Python installed. You can download it from the [official Python website](https://www.python.org/downloads/).

You can verify if Python is installed by running:

```bash
python --version
```

### 2. Pandoc
This script uses Pandoc to convert DOCX files to Markdown. You can download and install Pandoc from the [official Pandoc website](https://pandoc.org/installing.html).

After installation, verify Pandoc is installed by running:

```bash
pandoc --version
```

### 3. Required Python Packages
You'll need several Python libraries. You can install them using `pip`:

```bash
pip install python-docx pdf2docx
```

If you don't have `pip` installed, follow [this guide](https://pip.pypa.io/en/stable/installation/) to install it.

### 4. Create a Virtual Environment (optional but recommended)
Creating a virtual environment is a good practice to keep dependencies isolated.

To create a virtual environment, run:

```bash
python -m venv venv
```

To activate the virtual environment:

- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

Once the virtual environment is activated, install the required packages:

```bash
pip install python-docx pdf2docx
```

## Usage

Once the environment is set up, you can use the script to convert a PDF file to a Markdown file.

### 1. Run the Script

You can run the script with the following command:

```bash
python pdf2docx2md.py --input_pdf <path_to_pdf_file> --output_dir <path_to_save_output_files>
```

### Example:

```bash
python pdf2docx2md.py  --input_pdf "./input.pdf" --output_dir "./output"
```

This command will:

1. Convert the `input.pdf` file to `input.docx`
2. Remove the first three lines from the `input.docx`
3. Convert the modified `input.docx` to `input.md`
4. Save all files in the `./output` directory

### Output:

- `input.docx`: The DOCX file converted from PDF.
- `input.md`: The Markdown file converted from DOCX.
- `media/`: If there are images or media, they will be saved in this folder.

## Troubleshooting

### 1. Pandoc Not Found
If you encounter a `Pandoc not found` error, make sure Pandoc is installed and the installation path is added to your system's environment variables.

### 2. Python Packages Not Installed
If you encounter an error related to missing Python packages, ensure that `python-docx` and `pdf2docx` are installed by running:

```bash
pip install python-docx pdf2docx
```

### 3. Permission Issues
If you encounter permission issues while saving files, ensure you have the correct permissions to write to the specified output directory.

## License

This project is licensed under the MIT License.
