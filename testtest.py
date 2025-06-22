
my_name = "eren"

print(f"python learning {my_name}")
#python comment

#print(type(my_name))

#number = input("enter a number: ")

#print(f"given number is: {number}")

#inputlar her zaman string olarak alınıyor,
#sonradan değiştirmemek için direk input cast yapabilirsin

#number2 = int(input("enter another number: "))

#number = int(number)

##number += 1
#number2 += 1

#print(f"increased value of it is: {number} and other number {number2}")

#range() fonksiyonu series of integer only return ediyor

for i in range(5):
  print(i)
print("for loop is done")

start = 3
stop = 5

for i in range(start, stop):
  print(i)
print("for loop is done")

start = 0
stop = 10
step = 2 # step cannot be zero

for i in range(start, stop, step):
  print(i)
print("for loop is done")