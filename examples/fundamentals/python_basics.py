# 1. Variabel dan tipe data dasar
name = "Rina"          # str  : teks
age = 21               # int  : bilangan bulat
height = 1.62          # float: bilangan desimal
is_student = True      # bool : benar/salah
print(name, age, height, is_student)
print(type(name).__name__, type(age).__name__, type(height).__name__, type(is_student).__name__)

# 2. List: kumpulan nilai berurutan; indeks dimulai dari 0
scores = [80, 75, 90, 60]
print("\nscores        :", scores)
print("pertama/terakhir:", scores[0], scores[-1])
print("irisan [1:3]  :", scores[1:3], "(indeks 1 dan 2; indeks 3 tidak ikut)")
scores.append(95)
print("setelah append:", scores, "| jumlah elemen:", len(scores))

# 3. Dictionary: pasangan kunci -> nilai
student = {"name": name, "age": age, "scores": scores}
print("\nstudent['age'] :", student["age"])
print("kunci          :", list(student.keys()))

# 4. Perulangan (for) dan percabangan (if)
total = 0
for s in scores:
    total = total + s
mean = total / len(scores)
print("\ntotal %d, rata-rata %.1f" % (total, mean))
if mean >= 75:
    status = "lulus"
else:
    status = "belum lulus"
print("status:", status)


# 5. Fungsi: kode yang dapat dipakai ulang; parameter = masukan, return = keluaran
def average(values):
    return sum(values) / len(values)


print("\naverage(scores) =", average(scores))
print("average([10, 20]) =", average([10, 20]))

# 6. Kondisi boolean dan list comprehension (menyaring dengan satu baris)
print("\nscores >= 75 ?", [s >= 75 for s in scores])
passed = [s for s in scores if s >= 75]
print("nilai yang lulus:", passed)

# 7. Kesalahan umum: indeks di luar rentang
try:
    scores[10]
except IndexError as error:
    print("\nIndexError:", error)
