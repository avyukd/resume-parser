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


WORDS_LIST = {
    "Work": ["(Work|WORK)", "(Experience(s?)|EXPERIENCE(S?))", "(History|HISTORY)"],
    "Education": ["(Education|EDUCATION)", "(Qualifications|QUALIFICATIONS)"],
    "Skills": [
        "(Skills|SKILLS)",
        "(Proficiency|PROFICIENCY)",
        "LANGUAGE",
        "CERTIFICATION",
    ],
    "Projects": ["(Projects|PROJECTS)"],
    "Activities": ["(Leadership|LEADERSHIP)", "(Activities|ACTIVITIES)"],
}


def parse_resume(filepath):
    nlp = spacy.load('en_core_web_sm')
    text = textract.process(filepath).decode("utf-8")
    doc = nlp(text)
    number = findNumber(doc)
    email = findEmail(doc)
    name = findName(doc, os.path.basename(filepath))
    city = findCity(doc)
    categories = extractCategories(text)
    skills = extractSkills(doc)
    workAndEducation = findWorkAndEducation(
            categories, doc, text, name
        )
    
    return {
        "filepath": filepath,
        "name":name,
        "number":number,
        "email":email,
        "city":city,
        "skills":skills,
        "workAndEducation":workAndEducation
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

def findCity(doc) -> Optional[str]:
    counter = Counter()
    """
    Helper function to extract most likely City/Country from nlp doc
    :param doc: SpaCy Doc of text
    :return: str:city/country if found, else None
    """
    for ent in doc.ents:
        if ent.label_ == "GPE":
            counter[ent.text] += 1

    if len(counter) >= 1:
        return counter.most_common(1)[0][0]
    return None

def extractSkills(doc) -> List[str]:
    """
    Helper function to extract skills from spacy nlp text

    :param doc: object of `spacy.tokens.doc.Doc`
    :return: list of skills extracted
    """
    tokens = [token.text for token in doc if not token.is_stop]
    data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "skills.csv")
    )
    skills = list(data.columns.values)
    skillset = []
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams
    for token in doc.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def countWords(line: str) -> int:
    """
    Counts the numbers of words in a line
    :param line: line to count
    :return count: num of lines
    """
    count = 0
    is_space = False
    for c in line:
        is_not_char = not c.isspace()
        if is_space and is_not_char:
            count += 1
        is_space = not is_not_char
    return count

def extractCategories(text) -> Dict[str, List[Tuple[int, int]]]:
    """
    Helper function to extract categories like EDUCATION and EXPERIENCE from text
    :param text: text
    :return: Dict[str, List[Tuple[int, int]]]: {category: list((size_of_category, page_count))}
    """
    data = defaultdict(list)
    page_count = 0
    prev_count = 0
    prev_line = None
    prev_k = None
    for line in text.split("\n"):
        line = re.sub(r"\s+?", " ", line).strip()
        for (k, wl) in WORDS_LIST.items():
            # for each word in the list
            for w in wl:
                # if category has not been found and not a very long line
                # - long line likely not a category
                if countWords(line) < 10:
                    match = re.findall(w, line)
                    if match:
                        size = page_count - prev_count
                        # append previous
                        if prev_k is not None:
                            data[prev_k].append((size, prev_count, prev_line))
                        prev_count = page_count
                        prev_k = k
                        prev_line = line
        page_count += 1

    # last item
    if prev_k is not None:
        size = page_count - prev_count - 1 # -1 cuz page_count += 1 on prev line
        data[prev_k].append((size, prev_count, prev_line))

    # choose the biggest category (reduce false positives)
    for k in data:
        if len(data[k]) >= 2:
            data[k] = [max(data[k], key=lambda x: x[0])]
    return data

def findWorkAndEducation(categories, doc, text, name) -> Dict[str, List[str]]:
        inv_data = {v[0][1]: (v[0][0], k) for k, v in categories.items()}
        line_count = 0
        exp_list = defaultdict(list)
        name = name.lower()

        current_line = None
        is_dot = False
        is_space = True
        continuation_sent = []
        first_line = None
        unique_char_regex = "[^\sA-Za-z0-9\.\/\(\)\,\-\|]+"

        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            match = re.search(r"^.*:", line)
            if match:
                line = line[match.end() :].strip()

            # get first non-space line for filtering since
            # sometimes it might be a page header
            if line and first_line is None:
                first_line = line

            # update line_countfirst since there are `continue`s below
            line_count += 1
            if (line_count - 1) in inv_data:
                current_line = inv_data[line_count - 1][1]
            # contains a full-blown state-machine for filtering stuff
            elif current_line == "Work":
                if line:
                    # if name is inside, skip
                    if name == line:
                        continue
                    # if like first line of resume, skip
                    if line == first_line:
                        continue
                    # check if it's not a list with some unique character as list bullet
                    has_dot = re.findall(unique_char_regex, line[:5])
                    # if last paragraph is a list item
                    if is_dot:
                        # if this paragraph is not a list item and the previous line is a space
                        if not has_dot and is_space:
                            if line[0].isupper() or re.findall(r"^\d+\.", line[:5]):
                                exp_list[current_line].append(line)
                                is_dot = False

                    else:
                        if not has_dot and (
                            line[0].isupper() or re.findall(r"^\d+\.", line[:5])
                        ):
                            exp_list[current_line].append(line)
                            is_dot = False
                    if has_dot:
                        is_dot = True
                    is_space = False
                else:
                    is_space = True
            elif current_line == "Education":
                if line:
                    # if not like first line
                    if line == first_line:
                        continue
                    line = re.sub(unique_char_regex, '', line[:5]) + line[5:]
                    if len(line) < 12:
                        continuation_sent.append(line)
                    else:
                        if continuation_sent:
                            continuation_sent.append(line)
                            line = " ".join(continuation_sent)
                            continuation_sent = []
                        exp_list[current_line].append(line)

        return exp_list

print(parse_resume("samples/resume1.pdf"))