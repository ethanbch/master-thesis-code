import csv
import os
import re

import PyPDF2


def extract_text_from_pdf(pdf_path):
    """Extrait tout le texte d'un fichier PDF."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        print(f"Erreur lors de la lecture de {pdf_path}: {e}")
    return text


def parse_and_write_stoxx_data(content, output_dir, original_filename):
    # 1. Date du fichier
    file_date_match = re.search(
        r"sl_sxxp_(\d{4})(\d{2})\.pdf", original_filename.lower()
    )
    if file_date_match:
        year = file_date_match.group(1)
        month = file_date_match.group(2)
        filename_date = f"{year}{month}01"
    else:
        filename_date = "UNKNOWN_DATE_01"

    output_filename = f"slpublic_sxxp_{filename_date}.csv"
    output_filepath = os.path.join(output_dir, output_filename)

    # 2. Date pour la colonne
    date_text_match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", content)
    if date_text_match:
        d_day, d_month, d_year = date_text_match.groups()
        creation_date = f"{d_year}{d_month}{d_day}"
    else:
        creation_date = filename_date if file_date_match else ""

    index_info = {"STOXX Europe 600": {"Symbol": "SXXP", "ISIN": "EU0009658202"}}

    # Correction du nom de la bourse ici
    exchange_map = {
        ".CO": "NASDAQ Copenhagen",
        ".AS": "Euronext Amsterdam",
        ".VX": "Six Swiss Exchange",
        ".S": "Six Swiss Exchange",
        ".PA": "Euronext Paris",
        ".L": "London SE",
        ".DE": "Xetra",
        ".MC": "Bolsa de Madrid",
        ".MI": "Borsa Italiana",
        ".HE": "Nasdaq Helsinki",
        ".ST": "Nasdaq Stockholm",
    }

    index_name = "STOXX Europe 600"
    idx_symbol = index_info[index_name]["Symbol"]
    idx_isin = index_info[index_name]["ISIN"]

    # En-tête mis à jour (sans FF Mcap)
    headers = [
        "Creation_Date",
        "Index_Symbol",
        "Index_Name",
        "Index ISIN",
        "Internal_Key",
        "ISIN",
        "RIC",
        "Instrument_Name",
        "Country",
        "Currency",
        "Exchange",
        "Index Membership",
        "Rank (FINAL)",
        "Rank (PREVIOUS)",
        "Comment",
        "Rank 2 (FINAL)",
        "Rank 2 (PREVIOUS)",
    ]

    with open(output_filepath, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(headers)

        isin_pattern = r"\b([A-Z]{2}[A-Z0-9]{10})\b"
        parts = re.split(isin_pattern, content)

        lignes_ajoutees = 0

        for i in range(1, len(parts) - 1, 2):
            isin = parts[i]
            block = parts[i + 1].replace("\n", " ")

            curr_match = re.search(r"\b(EUR|GBP|CHF|DKK|SEK|NOK|CZK|PLN|HUF)\b", block)
            currency = curr_match.group(1) if curr_match else ""

            country_match = re.search(
                r"\b(GB|FR|DE|CH|NL|SE|DK|ES|IT|FI|BE|NO|AT|PT|IE|LU|GR|CZ)\b", block
            )
            country = country_match.group(1) if country_match else ""

            memb_match = re.search(r"\b(Large|Mid|Small)\b", block, re.IGNORECASE)
            membership = memb_match.group(1).capitalize() if memb_match else ""

            ric_match = re.search(r"\b([A-Za-z0-9]+\.[A-Za-z0-9]+)\b", block)
            ric = ric_match.group(1) if ric_match else ""

            int_key_match = re.search(r"\b(\d{6})\b", block)
            internal_key = int_key_match.group(1) if int_key_match else ""

            # Extraction des rangs
            all_digits = re.findall(r"\b(\d{1,6})\b", block)
            if internal_key in all_digits:
                all_digits.remove(internal_key)

            # On ignore les grands nombres (comme le Mcap) pour les rangs
            ranks = [d for d in all_digits if len(d) <= 5]
            rank_final = (
                ranks[-2] if len(ranks) >= 2 else (ranks[-1] if len(ranks) == 1 else "")
            )
            rank_previous = ranks[-1] if len(ranks) >= 2 else ""

            exchange = ""
            if ric and "." in ric:
                exchange = exchange_map.get(ric[ric.index(".") :], "")

            # Nettoyage agressif du nom de l'instrument
            instrument_name = block

            # 1. On retire les éléments connus
            for item in [currency, country, membership, ric, internal_key]:
                if item:
                    instrument_name = instrument_name.replace(str(item), " ", 1)

            # 2. On retire les capitalisations boursières (ex: 195.9)
            instrument_name = re.sub(r"\b\d+\.\d+\b", "", instrument_name)

            # 3. On retire tous les blocs de chiffres isolés (comme les Sedols fragmentés ou les rangs)
            instrument_name = re.sub(r"\b\d+\b", "", instrument_name)

            # 4. Nettoyage de la ponctuation et des espaces multiples
            instrument_name = re.sub(r"[^A-Za-z\&\-\.\s]", "", instrument_name)

            # 5. On retire les lettres majuscules seules qui traînent à la fin (ex: "CH" ou " P ")
            # Note: On garde le "P" si c'est collé au mot ou pertinent, mais on nettoie les résidus
            instrument_name = re.sub(
                r"\b(CH|GB|FR|DE|NL)\b", "", instrument_name
            )  # Retire les codes pays restants
            instrument_name = re.sub(r"\s+", " ", instrument_name).strip()

            # Ligne de sortie sans la capitalisation boursière
            output_row = [
                creation_date,
                idx_symbol,
                index_name,
                idx_isin,
                internal_key,
                isin,
                ric,
                instrument_name,
                country,
                currency,
                exchange,
                membership,
                rank_final,
                rank_previous,
                "",
                "",
                "",
            ]

            writer.writerow(output_row)
            lignes_ajoutees += 1


def process_all_pdfs(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Erreur : Le dossier d'entrée '{input_dir}' n'existe pas.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Début de l'analyse dans : {input_dir}")

    pdf_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_count += 1
            file_path = os.path.join(input_dir, filename)
            print(f"Traitement de : {filename}...")

            raw_text = extract_text_from_pdf(file_path)
            if raw_text:
                parse_and_write_stoxx_data(raw_text, output_dir, filename)

    if pdf_count == 0:
        print("Aucun fichier PDF trouvé.")
    else:
        print("Opération terminée avec succès !")


# --- Paramètres à ajuster ---
DOSSIER_SOURCE = "./data/pdf_to_parse"
DOSSIER_SORTIE = "./data/raw/inclusions"

# Lancement
if __name__ == "__main__":
    process_all_pdfs(DOSSIER_SOURCE, DOSSIER_SORTIE)
