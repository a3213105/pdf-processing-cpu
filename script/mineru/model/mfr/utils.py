import re

LEFT_PATTERN = re.compile(r'(\\left)(\S*)')
RIGHT_PATTERN = re.compile(r'(\\right)(\S*)')
LEFT_COUNT_PATTERN = re.compile(r'\\left(?![a-zA-Z])')
RIGHT_COUNT_PATTERN = re.compile(r'\\right(?![a-zA-Z])')
LEFT_RIGHT_REMOVE_PATTERN = re.compile(r'\\left\.?|\\right\.?')

def fix_latex_left_right(s, fix_delimiter=True):
    """
    Fix in LaTeX\\leftand\\rightOrder
    1. Make sure they are followed by a valid delimiter
    2. balance\\leftand\\rightquantity
    """
    # Whitelist separator
    valid_delims_list = [r'(', r')', r'[', r']', r'{', r'}', r'/', r'|',
                         r'\{', r'\}', r'\lceil', r'\rceil', r'\lfloor',
                         r'\rfloor', r'\backslash', r'\uparrow', r'\downarrow',
                         r'\Uparrow', r'\Downarrow', r'\|', r'\.']

    # for\leftAdd dot after missing valid delimiter
    def fix_delim(match, is_left=True):
        cmd = match.group(1)  # \left or \right
        rest = match.group(2) if len(match.groups()) > 1 else ""
        if not rest or rest not in valid_delims_list:
            return cmd + "."
        return match.group(0)

    # Use more precise pattern matching\leftand\rightOrder
    # Make sure they are standalone commands and not part of other commands
    # Use precompiled regex and unified callback functions
    if fix_delimiter:
        s = LEFT_PATTERN.sub(lambda m: fix_delim(m, True), s)
        s = RIGHT_PATTERN.sub(lambda m: fix_delim(m, False), s)

    # Calculate more accurately\leftand\rightquantity
    left_count = len(LEFT_COUNT_PATTERN.findall(s))  # no match\lefteqnwait
    right_count = len(RIGHT_COUNT_PATTERN.findall(s))  # no match\rightarrowwait

    if left_count == right_count:
        # If the numbers are equal, check if they are in the same group
        return fix_left_right_pairs(s)
        # return s
    else:
        # If the numbers are not equal, remove all\leftand\right
        # logger.debug(f"latex:{s}")
        # logger.warning(f"left_count: {left_count}, right_count: {right_count}")
        return LEFT_RIGHT_REMOVE_PATTERN.sub('', s)


def fix_left_right_pairs(latex_formula):
    """
    Detect and repair LaTeX formulas\\leftand\\rightNot in the same group

    Args:
        latex_formula (str): Entered LaTeX formula

    Returns:
        str: Repaired LaTeX formula
    """
    # Used to track curly brace nesting levels
    brace_stack = []
    # for storage\leftinformation: (Location, depth, separator)
    left_stack = []
    # Storage needs to be adjusted\rightinformation: (start position, end position, target location)
    adjustments = []

    i = 0
    while i < len(latex_formula):
        # Check if it is an escape character
        if i > 0 and latex_formula[i - 1] == '\\':
            backslash_count = 0
            j = i - 1
            while j >= 0 and latex_formula[j] == '\\':
                backslash_count += 1
                j -= 1

            if backslash_count % 2 == 1:
                i += 1
                continue

        # Detection\leftOrder
        if i + 5 < len(latex_formula) and latex_formula[i:i + 5] == "\\left" and i + 5 < len(latex_formula):
            delimiter = latex_formula[i + 5]
            left_stack.append((i, len(brace_stack), delimiter))
            i += 6  # jump over\leftand delimiter
            continue

        # Detection\rightOrder
        elif i + 6 < len(latex_formula) and latex_formula[i:i + 6] == "\\right" and i + 6 < len(latex_formula):
            delimiter = latex_formula[i + 6]

            if left_stack:
                left_pos, left_depth, left_delim = left_stack.pop()

                # if\leftand\rightNot at the same brace depth
                if left_depth != len(brace_stack):
                    # turn up\leftThe end position of the brace group
                    target_pos = find_group_end(latex_formula, left_pos, left_depth)
                    if target_pos != -1:
                        # Records need to be moved\right
                        adjustments.append((i, i + 7, target_pos))

            i += 7  # jump over\rightand delimiter
            continue

        # Handle curly braces
        if latex_formula[i] == '{':
            brace_stack.append(i)
        elif latex_formula[i] == '}':
            if brace_stack:
                brace_stack.pop()

        i += 1

    # Apply adjustments, working from back to front to avoid index changes
    if not adjustments:
        return latex_formula

    result = list(latex_formula)
    adjustments.sort(reverse=True, key=lambda x: x[0])

    for start, end, target in adjustments:
        # extract\rightpart
        right_part = result[start:end]
        # Delete from original location
        del result[start:end]
        # Insert at target location
        result.insert(target, ''.join(right_part))

    return ''.join(result)


def find_group_end(text, pos, depth):
    """Find the end of a brace group of a specific depth"""
    current_depth = depth
    i = pos

    while i < len(text):
        if text[i] == '{' and (i == 0 or not is_escaped(text, i)):
            current_depth += 1
        elif text[i] == '}' and (i == 0 or not is_escaped(text, i)):
            current_depth -= 1
            if current_depth < depth:
                return i
        i += 1

    return -1  # No corresponding end position found


def is_escaped(text, pos):
    """Check if characters are escaped"""
    backslash_count = 0
    j = pos - 1
    while j >= 0 and text[j] == '\\':
        backslash_count += 1
        j -= 1

    return backslash_count % 2 == 1


def fix_unbalanced_braces(latex_formula):
    """
    Detect whether braces in LaTeX formulas are closed and remove braces that cannot be paired

    Args:
        latex_formula (str): Entered LaTeX formula

    Returns:
        str: LaTeX formula after removing unpairable curly braces
    """
    stack = []  # Index to store the left parenthesis
    unmatched = set()  # Store the index of unmatched brackets
    i = 0

    while i < len(latex_formula):
        # Check if it is an escaped curly brace
        if latex_formula[i] in ['{', '}']:
            # Count the number of preceding consecutive backslashes
            backslash_count = 0
            j = i - 1
            while j >= 0 and latex_formula[j] == '\\':
                backslash_count += 1
                j -= 1

            # If there is an odd number of backslashes before it, the brace is escaped and does not participate in the match.
            if backslash_count % 2 == 1:
                i += 1
                continue

            # Otherwise, the brace participates in the match
            if latex_formula[i] == '{':
                stack.append(i)
            else:  # latex_formula[i] == '}'
                if stack:  # There is a corresponding left bracket
                    stack.pop()
                else:  # There is no corresponding left bracket
                    unmatched.add(i)

        i += 1

    # All unmatched left parentheses
    unmatched.update(stack)

    # Build new string, removing unmatched brackets
    return ''.join(char for i, char in enumerate(latex_formula) if i not in unmatched)


def process_latex(input_string):
    """
        Handling backslashes in LaTeX formulas:
        1. if\followed by special characters (#$%&~_^\\{})or spaces, remain unchanged
        2. if\Followed by two lowercase letters, remain unchanged
        3. In other cases, in\add space after

        Args:
            input_string (str): Entered LaTeX formula

        Returns:
            str: Processed LaTeX formula
        """

    def replace_func(match):
        # get\后面的字符
        next_char = match.group(1)

        # If it is a special character or space, leave it unchanged
        if next_char in "#$%&~_^|\\{} \t\n\r\v\f":
            return match.group(0)

        # If it's a letter, check the next character
        if 'a' <= next_char <= 'z' or 'A' <= next_char <= 'Z':
            pos = match.start() + 2  # \xposition after
            if pos < len(input_string) and ('a' <= input_string[pos] <= 'z' or 'A' <= input_string[pos] <= 'Z'):
                # The next character is also a letter and remains unchanged
                return match.group(0)

        # In other cases, in\add space after
        return '\\' + ' ' + next_char

    # match\Followed by a character
    pattern = r'\\(.)'

    return re.sub(pattern, replace_func, input_string)

# Common mathematical environments available in KaTeX/MathJax
ENV_TYPES = ['array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix',
             'Bmatrix', 'Vmatrix', 'cases', 'aligned', 'gathered', 'align', 'align*']
ENV_BEGIN_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}') for env in ENV_TYPES}
ENV_END_PATTERNS = {env: re.compile(r'\\end\{' + env + r'\}') for env in ENV_TYPES}
ENV_FORMAT_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}\{([^}]*)\}') for env in ENV_TYPES}

def fix_latex_environments(s):
    """
    Detect the environment (such as array) in LaTeX\\beginand\\endDoes it match
    1. if missing\\begintags are added at the beginning
    2. if missing\\endtags are added at the end
    """
    for env in ENV_TYPES:
        begin_count = len(ENV_BEGIN_PATTERNS[env].findall(s))
        end_count = len(ENV_END_PATTERNS[env].findall(s))

        if begin_count != end_count:
            if end_count > begin_count:
                format_match = ENV_FORMAT_PATTERNS[env].search(s)
                default_format = '{c}' if env == 'array' else ''
                format_str = '{' + format_match.group(1) + '}' if format_match else default_format

                missing_count = end_count - begin_count
                begin_command = '\\begin{' + env + '}' + format_str + ' '
                s = begin_command * missing_count + s
            else:
                missing_count = begin_count - end_count
                s = s + (' \\end{' + env + '}') * missing_count

    return s


REPLACEMENTS_PATTERNS = {
    re.compile(r'\\underbar'): r'\\underline',
    re.compile(r'\\Bar'): r'\\hat',
    re.compile(r'\\Hat'): r'\\hat',
    re.compile(r'\\Tilde'): r'\\tilde',
    re.compile(r'\\slash'): r'/',
    re.compile(r'\\textperthousand'): r'‰',
    re.compile(r'\\sun'): r'☉',
    re.compile(r'\\textunderscore'): r'\\_',
    re.compile(r'\\fint'): r'⨏',
    re.compile(r'\\up '): r'\\ ',
    re.compile(r'\\vline = '): r'\\models ',
    re.compile(r'\\vDash '): r'\\models ',
    re.compile(r'\\sq \\sqcup '): r'\\square ',
    re.compile(r'\\copyright'): r'©',
}
QQUAD_PATTERN = re.compile(r'\\qquad(?!\s)')


def remove_up_commands(s: str):
    """Remove unnecessary up commands from LaTeX code."""
    UP_PATTERN = re.compile(r'\\up([a-zA-Z]+)')
    s = UP_PATTERN.sub(
        lambda m: m.group(0) if m.group(1) in ["arrow", "downarrow", "lus", "silon"] else f"\\{m.group(1)}", s
    )
    return s


def remove_unsupported_commands(s: str):
    """Remove unsupported LaTeX commands."""
    COMMANDS_TO_REMOVE_PATTERN = re.compile(
        r'\\(?:lefteqn|boldmath|ensuremath|centering|textsubscript|sides|textsl|textcent|emph|protect|null)')
    s = COMMANDS_TO_REMOVE_PATTERN.sub('', s)
    return s


def latex_rm_whitespace(s: str):
    """Remove unnecessary whitespace from LaTeX code."""
    s = fix_unbalanced_braces(s)
    s = fix_latex_left_right(s)
    s = fix_latex_environments(s)

    s = remove_up_commands(s)
    s = remove_unsupported_commands(s)

    # Apply all replacements
    for pattern, replacement in REPLACEMENTS_PATTERNS.items():
        s = pattern.sub(replacement, s)

    # Handling backslashes and spaces in LaTeX
    s = process_latex(s)

    # \qquadtrailing spaces
    s = QQUAD_PATTERN.sub(r'\\qquad ', s)

    # If the string ends with a backslash, remove the last backslash
    while s.endswith('\\'):
        s = s[:-1]

    return s