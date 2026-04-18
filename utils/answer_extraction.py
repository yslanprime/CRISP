"""Answer extraction helpers for math and reasoning datasets."""

import re

import regex


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _fix_sqrt(string):
    string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", string)
    return string


def _fix_tan(string):
    string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", string)
    return string


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    stripped_unit = re.sub(r"\\text{.*?}$", "", string).strip()
    if stripped_unit and stripped_unit != string:
        string = stripped_unit

    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()
    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("x\\in", "")
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    string = string.replace("\\cdot", "")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")
    string = re.sub(r"\\mbox{.*?}", "", string)
    string = string.replace("'", "")
    string = string.replace("\"", "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    string = regex.sub(r"(\\|,|\.)+$", "", string)
    return string


def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        depth = 0
        for i, char in enumerate(piece):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_program_output(pred_str):
    if "```output" not in pred_str:
        return ""
    pred_str = pred_str.split("```output")[-1]
    if "```" in pred_str:
        pred_str = pred_str.split("```")[0]
    return pred_str.strip()


def extract_answer(pred_str, exhaust=False):
    pred = []
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    elif "boxed" in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif "he answer is" in pred_str:
        pred = [pred_str.split("he answer is")[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output:
            pred.append(program_output)
        else:
            ans = re.findall(r"-?\d*\.?\d+", pred_str.replace(",", ""))
            pred.append(ans[-1] if ans else "")

    cleaned = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":").rstrip(".").rstrip("/")
        cleaned.append(strip_string(ans))

    if exhaust:
        return cleaned
    return cleaned[-1] if cleaned else ""


def extract_math_answer(question, reasoning, task):
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if "separated by commas" in question and all(ch not in ans for ch in "()[]"):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")])
        else:
            answer.append(ans.strip())
    return answer
