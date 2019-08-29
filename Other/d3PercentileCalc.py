import math
import subprocess

# subprocess.call([r'C:\Users\ericw\Desktop\Programming\python\Other\Matrix.bat'])

#preformat input data via .bat
subprocess.call([r"C:\Users\ericw\Desktop\Programming\python\Other\preformat.bat"])
numbers = open("C:/Users/ericw/Desktop/Programming/python/Other/d3Formatted.txt", "r+")
numb = numbers.read()
# #split by , and remove all empty values
print("Len:", len(numb))
numb = list(filter(None, numb.split(" ")))
print("Len2:", len(numb))
# print(numb)
numb = [float(x) for x in numb]
numb = sorted(numb)

# set file pointer to 0 so windows doesn't create /x00 placeholders
numbers.seek(0)
# delete text in file from cursor pt 0
numbers.truncate(0)

# repopulate with ordered data
for i in range(len(numb)):
    numbers.write(str(numb[i]))
    if i != len(numb):
        numbers.write(",")

numbers.close()

def findPercentile(percentile, data):
    """Find value of given percentile given dataset (def3 percentile)
        percentile given as integer value, such as 50, for 50%
        data should be type array, sorted small to large
    """
    # R=P/100 * (N+1)
    R = (percentile / 100) * (len(data) + 1)
    # R -> IR and FR  etc 10.25 -> 10 and .25
    FR, IR = math.modf(R)
    IR = int(IR)
    print(percentile / 100, "%")
    print(len(data) + 1, "N+1")
    print(f"IR:{IR}, FR:{FR}")
    print(R)
    if FR == 0:
        print(f"FR=0, {percentile}% of the data = {data[IR-1]}")
    else:
        print(f"{percentile}% of the data = {abs(data[IR]-data[IR+1])*FR+data[IR]}")


findPercentile(25, numb)
