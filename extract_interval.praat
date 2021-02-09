#asks for a filename, interval (start & end time), location (path)
#returns a tab seperated file of all labels in the tiers, with start and end point

# input fn001346.wav 1.00 2.99 [/Users/martijnbentum/] (optional)
#output fn001346_start_end.wav

form get filename
	word wav fn001346.wav
	positive start 1.5
	positive end 2.5
	word path /vol/tensusers/mbentum/FRYSIAN_ASR/TABLES/
	word label 
	
endform

writeInfoLine: "PATH: ", path$, "FILENAME: ", wav$
appendInfoLine: "Extracting sound from: ", start, " to: ", end
wav_id = Read from file: path$ + wav$
fn$ = selected$ ("Sound")

part_id = Extract part: start, end, "rectangular", 1, "no"
Save as WAV file: "'path$''fn$'_'start'_'end'_'label$'.wav"

#Edit cannot be used in commmand line mode!
#Edit
#editor Sound 'fn$'
#	Select... start end
#	Move start of selection to nearest zero crossing
#	Move end of selection to nearest zero crossing
#	Save selected sound as WAV file... 'path$''fn$' _'output$'.wav
#Close
#---

appendInfoLine: "File written to:  ", path$, fn$, "_", output$, "_", start, "_", end, ".wav"
#fn$,"_",output$,".wav'

