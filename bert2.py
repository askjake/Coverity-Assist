import os
import sys
import json
import paramiko
import requests
import numpy as np
import psycopg2
from psycopg2 import sql
import hashlib
import stat
import argparse
from threading import Thread
from transformers import GPT2TokenizerFast
from bertopic import BERTopic
import pandas as pd
import time 

# Directory for saving visualizations
OUTPUT_DIR = "bert"

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def split_content_into_chunks(content, max_tokens=1024, overlap=50):
    """
    Split content into chunks with specified overlap and max tokens per chunk.
    """
    input_tokens = tokenizer.encode(content)
    chunks = []
    start = 0
    while start < len(input_tokens):
        end = start + max_tokens
        chunk_tokens = input_tokens[start:end]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
        start += max_tokens - overlap  # Ensure overlap between chunks
    return chunks
    
def load_credentials(credentials_file):
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
        return credentials
    except Exception as e:
        print(f"Error loading credentials from {credentials_file}: {e}")
        return None

def ssh_connect(host, username, password):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=username, password=password)
        return ssh
    except Exception as e:
        print(f"SSH connection failed: {e}")
        return None

def sftp_list_files(sftp, remote_path, file_filter=None):
    """
    List files in a directory, optionally filtering by a substring in the filename.
    """
    all_files = []

    def recursive_list(path):
        try:
            for entry in sftp.listdir_attr(path):
                full_path = os.path.join(path, entry.filename)
                if stat.S_ISDIR(entry.st_mode):
                    recursive_list(full_path)
                elif file_filter in entry.filename:
                    all_files.append(full_path)
        except Exception as e:
            print(f"Error accessing {path}: {e}")

    recursive_list(remote_path)
    return all_files

def read_remote_file(sftp, remote_file_path):
    try:
        with sftp.open(remote_file_path, 'r') as remote_file:
            content = remote_file.read()
        return content
    except Exception as e:
        print(f"Failed to read remote file {remote_file_path}: {e}")
        return None

def get_embedding(chunk_content, api_url, expected_size, model="dolphin-mixtral"):
    """
    Fetch the embedding for a given chunk.
    """
    try:
        payload = {
            "model": model,
            "prompt": chunk_content,
            "options": {"num_ctx": 32768},
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, json=payload, headers=headers, timeout=240)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")

        if embedding and len(embedding) == expected_size:
            return np.array(embedding, dtype=np.float32)
        else:
            print(f"Warning: Invalid embedding size returned ({len(embedding) if embedding else 'None'})")
            return None
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return None
        
def get_content_from_db(content_hashes, db_config, embedding_table):
    """Fetch plain text content from the database using hash codes."""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        placeholders = ', '.join(['%s'] * len(content_hashes))

        # Ensure the table name preserves its case by quoting it
        query = sql.SQL("""
            SELECT content_hash, original_content 
            FROM {table}
            WHERE content_hash IN ({placeholders})
        """).format(
            table=sql.Identifier(embedding_table),
            placeholders=sql.SQL(placeholders)
        )

        cursor.execute(query, tuple(content_hashes))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Convert to dictionary and ensure all content is a string
        result = {}
        fallback_count = 0
        for content_hash, original_content in rows:
            if original_content is not None:
                result[content_hash] = str(original_content)
            else:
                #print(f"Warning: No content found for hash {content_hash}. Using hash as fallback.")
                result[content_hash] = content_hash  # Fallback to hash if content is missing
                fallback_count += 1

        return result, fallback_count
    except Exception as e:
        print(f"Error fetching content from database: {e}")
        return {}, len(content_hashes)  # Assume all were fallback if an error occu

def fetch_embeddings_from_db(db_config, embedding_table, expected_size=768):
    """
    Fetch embeddings from the specified table and return them along with their content hashes.
    """
    skipped_count = 0
    embeddings = []
    content_hashes = []

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        fetch_query = sql.SQL("""
            SELECT content_hash, embedding
            FROM {table};
        """).format(table=sql.Identifier(embedding_table))

        cursor.execute(fetch_query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        for content_hash, embedding in rows:
            #content_hash, embedding = rows
            if isinstance(embedding, list) and len(embedding) == expected_size:  # Expected size
                embeddings.append(np.array(embedding, dtype=np.float32))
                content_hashes.append(content_hash)
            else:
                print(f"Skipping invalid embedding for content_hash {content_hash}")
                skipped_count += 1

        print(f"Skipped {skipped_count} invalid  {expected_size} mismatched embeddings.")
        if embeddings:
            return embeddings, content_hashes  # Return both
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching embeddings sized from table {embedding_table}: {e}")
        return None, None

def fetch_embeddings_in_batches(db_config, embedding_table, expected_size=768, batch_size=10000):
    """
    Fetch embeddings from the specified table in batches.
    Yields batches of embeddings and content_hashes.
    """
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(name='embeddings_cursor')  # Server-side cursor
        fetch_query = sql.SQL("""
            SELECT content_hash, embedding
            FROM {table};
        """).format(table=sql.Identifier(embedding_table))
        cursor.execute(fetch_query)

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            embeddings = []
            content_hashes = []
            for content_hash, embedding in rows:
                if isinstance(embedding, list) and len(embedding) == expected_size:
                    embeddings.append(np.array(embedding, dtype=np.float32))
                    content_hashes.append(content_hash)
                else:
                    print(f"Skipping invalid embedding for content_hash {content_hash}")
            yield embeddings, content_hashes
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error fetching embeddings from table {embedding_table}: {e}")
        yield None, None


def create_embeddings_table(db_config, embedding_table):
    """
    Create table for embeddings if it doesn't already exist, and add original_content column if missing.
    """
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        # Create the table if it doesn't exist
        cursor.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                content_hash TEXT NOT NULL,
                chunk_index INT NOT NULL,
                chunk_content TEXT NOT NULL,
                original_content TEXT NOT NULL,
                embedding REAL[] NOT NULL,
                UNIQUE (content_hash, chunk_index)
            );
        """).format(table=sql.Identifier(embedding_table)))
        conn.commit()

        # Check if 'original_content' column exists
        cursor.execute(sql.SQL("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = {table_name} AND column_name = 'original_content';
        """).format(table_name=sql.Literal(embedding_table)))

        if cursor.fetchone() is None:
            # 'original_content' column does not exist, so add it
            cursor.execute(sql.SQL("""
                ALTER TABLE {table}
                ADD COLUMN original_content TEXT;
            """).format(table=sql.Identifier(embedding_table)))
            conn.commit()
            print(f"Added 'original_content' column to table {embedding_table}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating or modifying table {embedding_table}: {e}")

def save_chunk_to_db(db_config, embedding_table, content_hash, total_chunks, chunk_index, chunk_content, original_content, embedding):
    """
    Save the embedding and content chunk to the specified table in the database.
    """
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        insert_query = sql.SQL("""
            INSERT INTO {table} (content_hash, chunk_index, chunk_content, original_content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (content_hash, chunk_index)
            DO UPDATE SET
                chunk_content = EXCLUDED.chunk_content,
                original_content = EXCLUDED.original_content,
                embedding = EXCLUDED.embedding;
        """).format(table=sql.Identifier(embedding_table))

        embedding_list = embedding.tolist()
        cursor.execute(insert_query, (content_hash, chunk_index, chunk_content, original_content, embedding_list))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully saved chunk {chunk_index}/{total_chunks} to {embedding_table}.")
    except Exception as e:
        print(f"Error saving chunk {chunk_index} to table {embedding_table}: {e}")
        print(f"SQL Query: {insert_query.as_string(conn)}")
        print(f"Values: {content_hash}, {chunk_index}, {chunk_content}, {original_content}, [embedding array]")

def save_chunks_to_db(db_config, embedding_table, chunks_data):
    """
    Save multiple chunks to the database in a batch.
    """
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO {table} (content_hash, chunk_index, chunk_content, original_content, embedding)
            VALUES %s
            ON CONFLICT (content_hash, chunk_index)
            DO UPDATE SET
                chunk_content = EXCLUDED.chunk_content,
                original_content = EXCLUDED.original_content,
                embedding = EXCLUDED.embedding;
        """).format(table=sql.Identifier(embedding_table))

        data_to_insert = [
            (
                chunk['content_hash'],
                chunk['chunk_index'],
                chunk['chunk_content'],
                chunk['original_content'],
                chunk['embedding'].tolist()
            )
            for chunk in chunks_data
        ]

        psycopg2.extras.execute_values(
            cursor, insert_query.as_string(conn), data_to_insert, template=None, page_size=100
        )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully saved {len(chunks_data)} chunks to {embedding_table}.")
    except Exception as e:
        print(f"Error saving chunks to table {embedding_table}: {e}")

def process_file(content_str, api_url, db_config, embedding_table, expected_size=768, max_tokens=2048, overlap=50, model="Alex"):
    """
    Process a single file's content: split into chunks, embed, and save to the database.
    """
    chunks = split_content_into_chunks(content_str, max_tokens, overlap)
    for chunk_index, chunk_content in enumerate(chunks):
        content_hash = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()
        embedding = get_embedding(chunk_content, api_url, expected_size, model=model)

        if embedding is not None:
            save_chunks_to_db(db_config, embedding_table, chunks_data)
            #save_chunk_to_db(db_config, embedding_table, content_hash, len(chunks), chunk_index, chunk_content, chunk_content, embedding)
        else:
            print(f"Skipping invalid embedding for chunk {chunk_index}")

def process_directory(ssh, api_url, db_config, embedding_table, directory, model, file_filter="", expected_size=768):
    """
    Process all files in a directory.
    """
    embedding_table = f"{embedding_table}_{model}"  # Append model name to table
    create_embeddings_table(db_config, embedding_table)

    sftp = ssh.open_sftp()
    remote_path = f"/ccshare/logs/smplogs/{directory}"
    print(f"Processing directory: {remote_path}")
    files = sftp_list_files(sftp, remote_path, file_filter=file_filter)

    for file_path in files:
        print(f"Processing file: {file_path}")
        content = read_remote_file(sftp, file_path)
        if content:
            content_str = content.decode("utf-8", errors="replace").replace("\x00", "")
            process_file(content_str, api_url, db_config, embedding_table, expected_size, model=model)
    sftp.close()

def visualize_with_bertopic(embedding_batches, topic_model, embedding_table, model, db_config):
    """
    Visualize embeddings with BERTopic, processing all data at once if possible.
    """
    all_embeddings = []
    all_documents = []

    for embeddings, content_hashes in embedding_batches:
        if len(embeddings) < 3:
            print(f"Skipping batch with less than 3 valid data points.")
            continue

        embeddings_array = np.array(embeddings, dtype=np.float32)
        content_mapping, _ = get_content_from_db(content_hashes, db_config, embedding_table)
        documents = [content_mapping.get(h, h) for h in content_hashes]

        all_embeddings.append(embeddings_array)
        all_documents.extend(documents)

    if not all_embeddings:
        print("No valid embeddings to visualize.")
        return

    all_embeddings = np.vstack(all_embeddings)

    # Fit BERTopic on all data
    topics, probs = topic_model.fit_transform(all_documents, all_embeddings)

    # Save visualizations
    topics_html = os.path.join(OUTPUT_DIR, f"{embedding_table}_{model}_topics.html")
    barchart_html = os.path.join(OUTPUT_DIR, f"{embedding_table}_{model}_barchart.html")
    topic_model.visualize_topics().write_html(topics_html)
    topic_model.visualize_barchart().write_html(barchart_html)

    print(f"Saved visualizations to {topics_html} and {barchart_html}.")

def main():
    parser = argparse.ArgumentParser(description="Process directories to embed files.")
    parser.add_argument("directories", nargs="*", help="Directories to process")
    parser.add_argument("-t", "--table", default="embeddings", help="Name of the embeddings table")
    parser.add_argument("-i", "--ip", default="10.79.85.43", help="API IP address")
    parser.add_argument("-p", "--port", default="11434", help="API IP port")
    parser.add_argument("-es", "--expected_size", default=768, help="expected table size")
    parser.add_argument("-m", "--model", default="dolphin-mixtral", help="Embedding model to use")
    parser.add_argument("-f", "--file", default="", help="Filter for files containing this substring")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize topics using BERTopic")
    parser.add_argument("-s", "--save", action="store_true", help="Save visualizations as HTML files")
    args = parser.parse_args()

    credentials = load_credentials("/home/montjac/JAMbot/credentials.txt")
    if not credentials:
        return

    db_config = {
        "host": "10.79.85.43",
        "dbname": credentials.get("DB_NAME", "chatbotdb"),
        "user": credentials.get("DB_USER"),
        "password": credentials.get("DB_PASSWORD"),
    }

    if args.visualize:
        import cuml
        from cuml.manifold import UMAP

        embedding_table = f"{args.table}_{args.model}"  # Use model-specific table
        embeddings, content_hashes = fetch_embeddings_from_db(db_config, embedding_table, args.expected_size)
        embeddings = fetch_embeddings_in_batches(db_config, embedding_table, expected_size=768, batch_size=10000)
        
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings found in the database.")
            return


        topic_model = BERTopic(umap_model=UMAP(
            n_neighbors=min(15, len(embeddings) - 1),  # Ensure n_neighbors < number of samples
            min_dist=0.1,
            metric='cosine'
        ))

        visualize_with_bertopic_in_batches(
            embeddings, content_hashes, topic_model, embedding_table, args.model, db_config, batch_size=100000
        )
        return

    if not args.directories:
        print("Error: No directories specified.")
        return

    ssh_host = credentials.get("linux_pc")
    ssh_username = credentials.get("username")
    ssh_password = credentials.get("password")
    api_url = f"http://{args.ip}:{args.port}/api/embeddings"

    ssh = ssh_connect(ssh_host, ssh_username, ssh_password)
    if not ssh:
        return

    threads = []
    for directory in args.directories[:4]:
        thread = Thread(target=process_directory, args=(ssh, api_url, db_config, args.table, directory, args.model, args.file, args.expected_size))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    ssh.close()
    print("Directory processing completed. Use --visualize to analyze topics.")

if __name__ == "__main__":
    main()