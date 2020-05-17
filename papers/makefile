final:
	pdflatex final_report
	-@bibtex final_report
	# need to run twice to get references
	pdflatex final_report
progress_tex:
	pdflatex progress_report.tex
clean:
	-@rm *.pdf
	-@rm *.aux *.out *.log
	-@rm *.blg *.bbl
