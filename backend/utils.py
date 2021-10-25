import re
import pandas as pd
import os
import sys
import spacy 
from collections import Counter, defaultdict
from datetime import datetime
from dateutil import relativedelta
from typing import *

import textract


def parse_resume(filepath):
    nlp = spacy.load('en_core_web_sm')
    text = textract.process(filepath).decode("utf-8")
    doc = nlp(text)
    number = findNumber(doc)
    email = findEmail(doc)
    name = findName(doc, os.path.basename(filepath))
    return {
        "name":name,
        "number":number,
        "email":email
    }
    
def findNumber(doc) -> Optional[str]:
    """
    Helper function to extract number from nlp doc
    :param doc: SpaCy Doc of text
    :return: int:number if found, else None
    """
    for sent in doc.sents:
        num = re.findall(r"\(?\+?\d+\)?\d+(?:[- \)]+\d+)*", sent.text)
        if num:
            for n in num:
                if len(n) >= 8 and (
                    not re.findall(r"^[0-9]{2,4} *-+ *[0-9]{2,4}$", n)
                ):
                    return n
    return None

def findEmail(doc) -> Optional[str]:
    """
    Helper function to extract email from nlp doc
    :param doc: SpaCy Doc of text
    :return: str:email if found, else None
    """
    for token in doc:
        if token.like_email:
            return token.text
    return None

def findName(doc, filename) -> Optional[str]:
    """
    Helper function to extract name from nlp doc
    :param doc: SpaCy Doc of text
    :param filename: used as backup if NE cannot be found
    :return: str:NAME_PATTERN if found, else None
    """
    to_chain = False
    all_names = []
    person_name = None

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if not to_chain:
                person_name = ent.text.strip()
                to_chain = True
            else:
                person_name = person_name + " " + ent.text.strip()
        elif ent.label_ != "PERSON":
            if to_chain:
                all_names.append(person_name)
                person_name = None
                to_chain = False
    if all_names:
        return all_names[0]
    else:
        try:
            base_name_wo_ex = os.path.splitext(os.path.basename(filename))[0]
            return base_name_wo_ex + " (from filename)"
        except:
            return None
