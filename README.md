# Agentic Search Engine for Investigative Journalism

This project is focused on building an agentic search engine tailored for investigative journalism, specifically targeting Chicago's Tax Increment Financing (TIF) data. The system is designed to handle both structured and unstructured data, providing a natural language interface to facilitate seamless querying.

## Project Goals

- **Integrate LlamaIndex for Qualitative Data**: 
  - Ingest and process PDFs containing TIF information.
  - Chunk and index these documents using OpenAI embeddings.
  - Enable semantic querying through a language model pipeline.

- **Route Queries Intelligently**:
  - Use a query planner to direct queries to either the structured SQL database or the unstructured PDF index based on the query's intent.

## Key Features

- **PDF Document Processing**:
  - `pdf_index/loader.py`: Load and chunk PDF documents for indexing.
  - `pdf_index/vector_index.py`: Build and persist a vector index from the processed PDFs.
  - `pdf_index/query_pdf.py`: Query the indexed PDFs using natural language.

- **Structured Data Querying**:
  - Utilize SQL queries on a DuckDB database to access structured TIF expenditure data.

## Usage Instructions

1. **Environment Setup**:
   - Ensure you have a `.env` file with your OpenAI API key set as `OPENAI_API_KEY`.

2. **PDF Indexing**:
   - Store your source PDF documents in the `pdfs/` folder.
   - Run `pdf_index/vector_index.py` to build the vector index from these documents.

3. **Querying**:
   - Use `query_pdf_index` in `pdf_index/query_pdf.py` to perform semantic queries on the PDF index.
   - Structured queries can be executed against the DuckDB database using the SQL interface.

4. **Future Enhancements**:
   - Implement `query_planner.py` to intelligently route queries between the PDF index and the SQL database.

## Requirements

- Python 3.7+
- OpenAI Python client
- Pandas
- dotenv
- DuckDB

## Getting Started

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Set up your environment variables in a `.env` file.
4. Follow the usage instructions to build indexes and perform queries.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- OpenAI for providing the language models and embeddings.
- Contributors to the open-source libraries used in this project.

