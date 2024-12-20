import os
from Bio import SeqIO
import multiprocessing
import threading

# Genetic code dictionary from main.py
genetic_code = {
    "F": ["UUU", "UUC"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "I": ["AUU", "AUC", "AUA"],
    "M": ["AUG"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "Y": ["UAU", "UAC"],
    "*": ["UAA", "UAG", "UGA"],
    "H": ["CAU", "CAC"],
    "Q": ["CAA", "CAG"],
    "N": ["AAU", "AAC"],
    "K": ["AAA", "AAG"],
    "D": ["GAU", "GAC"],
    "E": ["GAA", "GAG"],
    "C": ["UGU", "UGC"],
    "W": ["UGG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
}


def protein_to_rna(protein_sequence):
    rna_sequences = []

    # Generate all possible RNA sequences for the given protein sequence
    def backtrack(index, current_rna):
        if index == len(protein_sequence):
            rna_sequences.append(current_rna)
            return
        amino_acid = protein_sequence[index]
        if amino_acid in genetic_code:
            for codon in genetic_code[amino_acid]:
                backtrack(index + 1, current_rna + codon)

    backtrack(0, "")
    return rna_sequences


def process_records(input_file, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, "r") as file:
        records = list(SeqIO.parse(file, "fasta"))
        total_records = len(records)
        records_per_process = total_records // 12  # Assuming 12 processes

        # Create a list of processes
        processes = []
        for i in range(12):  # Create 12 processes
            start_index = i * records_per_process
            end_index = start_index + records_per_process if i < 11 else total_records
            process = multiprocessing.Process(
                target=process_records_chunk,
                args=(records[start_index:end_index], output_folder),
            )
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()


def process_records_chunk(records_chunk, output_folder):
    for record in records_chunk:
        protein_sequence = str(record.seq).replace("*", "")  # Remove asterisks
        rna_sequences = protein_to_rna(protein_sequence)

        # Write only DNA sequences for this protein to a single output file
        output_file_path = os.path.join(
            output_folder, f"{record.id}_dna_sequences.fasta"
        )
        with open(output_file_path, "w") as output_file:
            for i, rna_sequence in enumerate(rna_sequences):
                # Convert RNA to DNA by replacing 'U' with 'T'
                dna_sequence = rna_sequence.replace("U", "T")
                output_file.write(f">{record.id}_dna_{i + 1}\n")
                output_file.write(f"{dna_sequence}\n")


def main():
    inputs = [
        (
            "AMPs_nonAMPs_data/AMPs/AMPs.fa",
            "final/AMPs",
        ),
        (
            "AMPs_nonAMPs_data/NonAMPs/NonAMPs.fa",
            "final/NonAMPs",
        ),
    ]

    threads = []
    for input_file, output_folder in inputs:
        thread = threading.Thread(
            target=process_records, args=(input_file, output_folder)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
