#!/usr/bin/env python3
"""
This has been written by Claude Sonnet 3.5.

Remove automatically inserted outputs from markdown files to prepare for knitr.
Removes:
- Code blocks with output (lines starting with ``` followed by content until next ```)
- Plot lines (![Plot](...) references)
"""

import sys
import re

def clean_markdown(content):
    lines = content.split('\n')
    cleaned_lines = []
    in_code_block = False
    code_block_is_output = False
    setup_added = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Add setup chunk after first heading
        if not setup_added and line.startswith('#'):
            cleaned_lines.append(line)
            cleaned_lines.append('')
            cleaned_lines.append("""```{r include=FALSE}
            knitr::knit_hooks$set(par = function(before, options, envir) {
  if (before && !is.null(options$par)) {
    par(options$par)
  }
})
knitr::opts_chunk$set(par = list(bty="l", pch=19))
            """)

            cleaned_lines.append('```')
            cleaned_lines.append('')
            setup_added = True
            i += 1
            continue

        # Check if this is a code block start
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting a code block
                in_code_block = True
                # Check if this is an R code block (keep) or output block (remove)
                if line.strip().startswith('```{r'):
                    # R code block - keep as is
                    code_block_is_output = False
                    cleaned_lines.append(line)
                else:
                    # Output block - skip it
                    code_block_is_output = True
            else:
                # Ending a code block
                in_code_block = False
                if not code_block_is_output:
                    cleaned_lines.append(line)
                code_block_is_output = False
        elif in_code_block:
            # Inside a code block
            if not code_block_is_output:
                cleaned_lines.append(line)
        elif line.strip().startswith('![Plot]'):
            # Skip plot references
            pass
        else:
            # Regular line
            cleaned_lines.append(line)

        i += 1

    return '\n'.join(cleaned_lines)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python clean_for_rmd.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, 'r') as f:
        content = f.read()

    cleaned_content = clean_markdown(content)
    print(cleaned_content)
