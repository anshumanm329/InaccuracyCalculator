#!/usr/bin/env python
# coding: utf-8

# # Расчёт неопределенности в математическом моделировании системы управления уровнем ГПН-ПИД
# 
# На данном проекте будет вычислена непорделленность в моделировании передаточной функции оюъекта управления, т.е., уровня жидкости для системы управления уровнем ГПН-ПИД.
# 
# * Математическое моделирование системы управления выполнятеся путем определения передаточной функции объекта управления и исполнительного механизма
# 
# * В данном случае,
#     * объект управления: уровень рабочей жидкости
#     * Исполнительный механизм: Сепаратор (который служит клапаном)
#     * Полученная передаточная функция: W(S) = 1,23/12,04S+1
#     * Параметры, влияющие на определения передаточной функции: Номинальная уровень (H), площадь сечения клапана (Sk) и площадь поверхности среды (В данном случае контейнера рабочей жидкости)
# 
# * По стандартом ГОСТ, известно что неопреденность в
#     * Измерении уровня: +/- 3мм
#     * Измерении площади: 5% измеренного значения 
# * Значения этих параметров: 
#     * Номинальная уровень: 700мм = 0,7м
#     * Площадь сечения клапана: 1,13 м2
#     * Площадь поверхности среды: 36 м2
# 
# 
# **Вычисление неопределенности в передаточной функции будет проведена путем вычисления неопределенности в определении коэфициента передачи (К) и в постоянной времени (Т), передаточной функции.**
# 
# **В этом проекте будет вычислена общая неопределлность в определении передаточной функции объекта управления методом Монте-Карло**
# 

# In[3]:


# Library imports
import numpy as np # numpy is required for random calculations and arrays
import pandas as pd # Pandas can be required if data structures and other plottings come into play
import math


# ## 1. Генерация массивов
# 

# In[5]:


# Put in a random seed 
np.random.seed(9)
# Generate an array of 1000 uniformly distributed random numbers between min. and max. values of nominal height
# Min. value of height = 690mm = 0.697m, max. value of height = 703mm = 0.703m
liquid_level_array = np.random.uniform(0.697, 0.703, 1000)
# Generate an array of 1000 uniformly distributed random numbers between min. and max. value of valve cross-section
# Min. and max. values of valve cross-section are: 1.0735m2 and 1.1865m2
valve_area_array = np.random.uniform(1.0735, 1.1865, 1000)
# Generate an array of 1000 uniformly distributed random numbers between min. and max. value of liquid tank surface area
# Min and max values are 34.2 m2 and 37.8 m2
tank_area_array = np.random.uniform(34.2, 37.8, 1000)


# ## 2. Вычисление неопределенности в коэффициенте передачи передаточной функции

# In[7]:


# Formula to calculate coefficient K: 2H/Sk (H=Nominal height, Sk= Valve cross-section)
# Define a function to calculate the error in measurement of coefficient 
def error_in_transfer_coeff(height_array, cross_section_array):
    """
    Function to calculate the error in measurement of transfer coefficient K. 
    Takes nominal height and valve cross-section area as input
    """
    # An array of 1000 zeroes
    k_vals = np.zeros(1000)
    # 1000 coefficient values for each value in the liquid level and valve area arrays
    for i in range(0,1000):
        k_vals[i] = ((2*height_array[i])/cross_section_array[i])
    # Now another array of 1000 zeroes
    k_vals_diff_array = np.zeros(1000)
    # Now calculate the absolute difference between the measured value and the various coefficient values in k_vals
    for j in range(0,1000):
        k_vals_diff_array[j]=abs(k_vals[j]-1.23)
    return(np.max(k_vals_diff_array))


# In[8]:


# Use this function with height array and valve area array for the parameters of GPN-PID
print(f"The error in measuring the transfer coefficient of the transfer function is {error_in_transfer_coeff(liquid_level_array,valve_area_array):.3f} ")


# ## 3. Вычисление неопределенности в определении посстоянной времени

# In[10]:


# Define a function to calculate the error in measuring the time constant
def error_in_time_constant(height_array, valve_area_array, tank_area_array):
    """
    Function to calculate the error in measuring the time constant of the transfunction of liquid level as a control object
    """
    # An array of 1000 zeroes to fill in values of 1000 time constants later
    t_vals = np.zeros(1000)
    # Calculate 1000 time constants from the values of values of the three arrays
    for i in range(0,1000):
        t_vals[i] = np.sqrt((2*height_array[i])/9.8) * ((tank_area_array[i])/(valve_area_array[i]))
    
    # Another array of zeroes is required to put in the differential values
    t_vals_diff_array = np.zeros(1000)
    # Now calculate the error 
    for j in range(0,1000):
        t_vals_diff_array[j] = abs(t_vals[j]-12.04)
    # Return the maximum value in the array of differentials
    return(np.max(t_vals_diff_array))
        
    


# In[11]:


# Calculate the error in time constant measurement for T/F in GPN-PID
print(f"The error in measuring the time constant of the transfer function of liquid level is {error_in_time_constant(liquid_level_array, valve_area_array, tank_area_array):.3f} s")


# ## Вычисление погрешности по методу Криновича

# * В данном случае, мы вычисляем погрешность в математическом моделировании системы управления уровнем нефти
# * Т.е. погрешность в определении передаточной функции
# * Передаточная функция управления уровнем используя клапан в качестве исполнительного механизма моделируется следующим образом:
#   `W(S) = K/1+TS`, где: К - Коэффициент передачи и Т - Постоянная времени
#   `К = 2Н/Sk`; Н - Номинальный уровень, Sk - Площадь сечения регулирующего клапана
#   `Т = sqrt(2H/g)*(Sk/Sп)`; g - ускорение силы тяжести, Sп - площадь поверхности среды (Бака)
# * В данном случае: `H = 700mm = 0.7m`
#                    `Sk = 1.13 m2`
#                    `Sп = 36 m2`
# * По ГОСТ: максимальная возможная погрешность в определении уровня = +/- 10мм,
#                                                                       `delta_H = 0.01m`
#                                               в определении площади: +/- 5% определеннного значения
#                                                                       `delta_Sk = 0.0565m2`
#                                                                       `delta_Sп = 1.8m2`
# * Потом вычислим параметр масштаба массива выходных значения `d_delta_k_array` и `d_delta_T_array`

# In[14]:


# Time to generate the random numbers as per Cauchy distribution
# N = 300
# scale parameter k = 0.001
np.random.seed(96)
# Generate an array of 300 uniformly distributed random numbers between 0 and 1
z = np.random.uniform(0,1,300)
k = 0.0001
H = 0.7
Sk = 1.13
Sp = 36
cauchy_height_array = np.zeros(300)
cauchy_valve_area_array = np.zeros(300)
cauchy_tank_area_array = np.zeros(300)
# fill the height array
for i in range(0,300):
    cauchy_height_array[i] = H + k*np.tan(math.pi*(z[i]-0.5))
# fill the valve area array
for i in range(0,300):
    cauchy_valve_area_array[i] = Sk + k*np.tan(math.pi*(z[i]-0.5))
# fill the tank area array
for i in range(0,300):
    cauchy_tank_area_array[i] = Sp + k*np.tan(math.pi*(z[i]-0.5))


# In[15]:


# Calculate time constant and transfer coefficient for each value of the three arrays
transfer_coefficient = 1.23
time_constant = 12.04

transfer_coefficient_array = np.zeros(300)
time_constant_array = np.zeros(300)
for i in range(0,300):
    transfer_coefficient_array [i] = ((2*cauchy_height_array[i])/(cauchy_valve_area_array[i]))
    time_constant_array[i] = ((np.sqrt(2*cauchy_height_array[i]/9.8))*(cauchy_tank_area_array[i]/cauchy_valve_area_array[i]))


# In[16]:


# Calculate the differentials
delta_k_array = np.zeros(300)
delta_T_array = np.zeros(300)
for i in range(0,300):
    delta_k_array[i] = abs(1.23-transfer_coefficient_array[i])
    delta_T_array[i] = abs(12.04-time_constant_array[i])


# In[17]:


np.max(delta_k_array), np.max(delta_T_array)


# In[18]:


def bisection_method_calculation (array, N, start_point, end_point):
    """
    Данная функция принимает массив (array) и размер массива (N) 
    и вычисляет sum(1-300)(d^2/d^2+del_yj^2) - N/2
    принимая значения start_point и end_point ддя d 
    """
    sum1=0
    sum2=0
    sum3=0
    if (end_point>start_point):
        #print("End point and start points are valid, we can proceed")
        
        mid_point =(start_point + (end_point - start_point)/2)
        for i in range (0,N):
            sum1 = sum1 + (start_point**2/(start_point**2+(array[i]**2)))
            sum2 = sum2 + (mid_point**2/(mid_point**2+(array[i]**2)))
            sum3 = sum3 + (end_point**2/(end_point**2+(array[i]**2)))
        
    else:
        print("Invalid start and end point. Please try again with different values")
    return(sum1-N/2, sum2-N/2, sum3-N/2)


# In[19]:


def bisection_method_logic(array, N, start_point, end_point):
    """
    В данной фунции, реализуется логика выполнения метода бисекции для наших распределенные по Коши массивов
    внутри функции выполняется функция bisection_method_calculation. 
    Из возрашенных результатов, если sum1-N/2, sum2-N/2, sum3-N/2 > 0 то невозможно получить результат в этом диапазоне и требуются
    другие начальные и конечные точки.

    Если один из них равен 0 то это решение

    Если sum1-N/2 < 0, sum2-N/2 > 0 и sum3-N/2 >0 то теперь решение надо найти между start-point и mid-point
    Если sum1-N/2 <0, sum2-N/2 < 0 и sum3-N/2 > 0 то решение надо найти между mid-point и end-point
    
    """
    start = start_point
    end = end_point
    mid =(start + (end - start)/2)
    (res1, res2, res3) = bisection_method_calculation(array, N, start, end)
    while (abs(res1)>=(10**-15) or abs(res2)>=(10**-15) or (abs(res3)>=(10**-15))):
        # None of the results are 0
        if (res1<0 and res2 >0 and res3 >0):
            # Sign change between start and mid points
            end = mid
            (res1, res2, res3) = bisection_method_calculation(array, N, start, end)
            mid = (start + (end - start)/2)
        elif (res1<0 and res2 < 0 and res3 >0):
            # Value changes between mid and end-point
            start = mid            
            (res1,res2,res3) = bisection_method_calculation(array, N, start, end)
            mid = (start + (end - start)/2)
        else:
            print("We have reached a break point and might have a solution")
            print(f"{(res1,res2,res3)}")
            print(f"{(start,mid,end)}")
            if (res1==0 or res2==0 or res3==0):
                print("We have a proper solution with one of the solutions being 0")
                if (res1==0):
                    print(f"Параметр масштаба: {start}")
                    return start
                elif(res2==0):
                    print(f"Параметр масштаба: {mid}")
                    return mid                    
                elif(res3==0):
                    print(f"Параметр масштаба: {end}")
                    return end
            else:
                print("We don't have a proper solution but we have a break point")
                if (np.min(res1,res2,res3)==res1):
                    print(f"Параметр масштаба: {start}")
                    return start
                elif (np.min(res1,res2,res3)==res2):
                    print(f"Параметр масштаба: {mid}")
                    return mid
                elif (np.min(res1,res2,res3)==res3):
                    print(f"Параметр масштаба: {end}")
                    return end
                break
                                    
            break
    else: 
        print("We might have a solution")
        print(f"{(res1,res2,res3)}")
        if (np.min(res1,res2,res3)==res1):
            print(f"Параметр масштаба: {start}")
            return start
        elif (np.min(res1,res2,res3)==res2):
            print(f"Параметр масштаба: {mid}")
            return mid
        elif (np.min(res1,res2,res3)==res3):
            print(f"Параметр масштаба: {end}")
            return end
    
        
        


# In[20]:


bisection_method_logic(delta_T_array, 300, 0, 1)


# In[21]:


bisection_method_logic(delta_k_array, 300, 0, 1)


# In[ ]:




