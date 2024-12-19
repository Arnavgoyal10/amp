import os
from Bio import SeqIO
import multiprocessing

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


def process_output_file(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, "r") as file:
        counter = 0
        for record in SeqIO.parse(file, "fasta"):
            if counter != 1:
                counter += 1
                continue

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
            "output/Drosophila_melanogaster/output_final.fasta",
            "final/Drosophila_melanogaster",
        ),
        # (
        #     "output/eastern_oyster/output_final.fasta",
        #     "final/eastern_oyster",
        # ),
        # ("output/Human/output_final.fasta", "final/Human"),
        # ("output/lion/output_final.fasta", "final/lion"),
        # ("output/zebra_fish/output_final.fasta", "final/zebra_fish"),
    ]

    with multiprocessing.Pool() as pool:
        pool.starmap(process_output_file, inputs)


if __name__ == "__main__":
    main()
