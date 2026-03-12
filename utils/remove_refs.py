
filename = r"c:\Users\nshadin\OneDrive - Kennesaw State University\QSplitFL\complete_rl_implementation\IEEE_QSplitFL_Sections.tex"
with open(filename, 'r') as f:
    lines = f.readlines()

# Lines are 0-indexed in python list.
# 740 in file (1-indexed) is index 739.
# 863 in file (1-indexed) is index 862.
# We want to keep 0 to 738, and 863 to end.

# Verify content visually if possible, but line numbers from view_file are reliable.
# line 740 content: "% REFERENCES\n"
# line 863 content: "\\end{thebibliography}\n"

start_idx = 739
end_idx = 862

print(f"Removing lines {start_idx+1} to {end_idx+1}")
print(f"Start content: {lines[start_idx]}")
print(f"End content: {lines[end_idx]}")

if "% REFERENCES" in lines[start_idx] and "\\end{thebibliography}" in lines[end_idx]:
    new_lines = lines[:start_idx] + lines[end_idx+1:]
    with open(filename, 'w') as f:
        f.writelines(new_lines)
    print("Successfully removed lines.")
else:
    print("Error: Line content mismatch. Aborting.")
