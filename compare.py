from scipy.fftpack import diff


expected_textfile = open("revised-ExpectedOutput.txt","r")
result_textfile = open("NNResults2.txt", "r")

expected_text = expected_textfile.read()
result_text = result_textfile.read()

errors = 0
differences = {}

for i in range(len(result_text)):
    if result_text[i] != expected_text[i]:
        errors = errors + 1
        if result_text[i] + "-" + expected_text[i] in differences:
            differences[result_text[i] + "-" + expected_text[i]] = (differences[result_text[i] + "-" + expected_text[i]] + 1)
        else:
            differences[result_text[i] + "-" + expected_text[i]] = 1


print(errors)
print(differences)
expected_textfile.close()
result_textfile.close()
