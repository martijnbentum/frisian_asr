#asks for an ort file and a location
#returns a tab seperated file of all labels in the tiers, with start and end point

# input fn001346.TextGrid
# output fn001346.table
Text writing preferences: "UTF-8"

form get filename
	word ort fn001346.ort
	word output_dir TABLES/
endform

# the above should result in default values to be loaded in ort input_dir etc.
# however when calling the script without arguement  ort input_dir output_dir are empty
# patched with the following


writeInfoLine: " FILENAME: " + ort$
writeInfoLine: " output_dir: " + output_dir$
ort_id = Read from file: ort$
fn$ = selected$ ("TextGrid")
writeInfoLine: " fn: " + fn$


table_id = Down to Table: "no", 2, "yes", "no"
selectObject: table_id
Save as tab-separated file: output_dir$+fn$ + ".Table"
