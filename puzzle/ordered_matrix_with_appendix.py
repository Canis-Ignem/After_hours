

bs = 4
main_list = []
aux_list = []
num_filas = 10
for i in range(num_filas-1):
    for j in range(bs):
        if j <= 1:
            aux_list.append(i + num_filas * j)
        else:
            aux_list.append( aux_list[-1]+num_filas-1 )
    main_list.append(aux_list)
    aux_list = []
main_list.append([num_filas-1])

for l in main_list:
    print(l)
    print()