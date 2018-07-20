from data_generator import generator

gen = generator(16)
l = round(5/2)
for item in gen:
    print(item[2])

print()
