# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:38:14 2019

@author: Carlos
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR #support vector machine
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics


def datos(beta, muestras, desviacion):    
    x = np.random.random(muestras)     
    e = np.random.randn(muestras) * desviacion
#    y = beta[0]*x+e
#    y = beta[0]*x+beta[1]*x**2+e
#    y = beta[0]*x+beta[1]*x**2+beta[2]*x**3+ e
    y=1/beta[0]*np.exp(x*x*x)+e
    return x.reshape((muestras,1)), y.reshape((muestras,1))
desviacion = 0.1
beta=[1,1,1]
n = 500
x, y = datos(beta, n, desviacion)


class ICONOS():
    
    def __init__(self):
        #modelo lineal
        posx,posy,separar=20,40,80
        self.fondo=(255,255,255)
        
        self.icon_lineal=cargar_iconos(posx,posy,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/linicon.png")[0]
        self.rect_lineal=cargar_iconos(posx,posy,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/linicon.png")[1]
        #modelo polinomial
        self.icon_poli=cargar_iconos(posx,posy+separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/policon.png")[0]
        self.rect_poli=cargar_iconos(posx,posy+separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/policon.png")[1]
        #modelo vectores de soporte
        self.icon_vector=cargar_iconos(posx,posy+2*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/vectoricon.png")[0]
        self.rect_vector=cargar_iconos(posx,posy+2*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/vectoricon.png")[1]
        #modelo arboles de decision
        self.icon_arbol=cargar_iconos(posx,posy+3*separar,"image/https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/arbolicon.png")[0]
        self.rect_arbol=cargar_iconos(posx,posy+3*separar,"image/https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/arbolicon.png")[1]
        #modelo bosques aleatorios
        self.icon_bosque=cargar_iconos(posx,posy+4*separar,"image/https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/bosqueicon.png")[0]
        self.rect_bosque=cargar_iconos(posx,posy+4*separar,"image/https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/bosqueicon.png")[1]
        #Redes elasticas
        self.icon_elastic=cargar_iconos(posx,posy+5*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/elasticon.png")[0]
        self.rect_elastic=cargar_iconos(posx,posy+5*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/elasticon.png")[1]
                       #PARAMETROS
        #aumentar grado(polinomio)
        self.icon_gradomas=cargar_iconos_grado(5*posx,posy+separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/gradicon.png")[0]
        self.rect_gradomas=cargar_iconos_grado(5*posx,posy+separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/gradicon.png")[1]
        #disminuir grado(polinomio)
        self.icon_gradomenos=cargar_iconos_grado(5*posx,posy+2*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/gradicon2.png")[0]
        self.rect_gradomenos=cargar_iconos_grado(5*posx,posy+1.4*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/gradicon2.png")[1]
        #modificar kernel
        self.icon_kernel=cargar_iconos(5*posx,posy+2*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/kernelicon.png")[0]
        self.rect_kernel=cargar_iconos(5*posx,posy+2*separar,"https://github.com/karloxkronfeld/Consola-Machine-Learning/blob/master/imag/kernelicon.png")[1]
        
        self.lista_modelos=[]        
        
    def click(self,modelo): 
        self.lista_modelos.append(modelo)       
              
    def dibujar(self,superficie):
        posx,posy,separar=20,40,80
       
        superficie.blit(self.icon_lineal,(posx,posy))
        superficie.blit(self.icon_poli,(posx,posy+separar))
        superficie.blit(self.icon_vector,(posx,posy+2*separar))
        superficie.blit(self.icon_arbol, (posx,posy+3*separar))
        superficie.blit(self.icon_bosque,(posx,posy+4*separar))
        superficie.blit(self.icon_elastic,(posx,posy+5*separar))
        
    def parametros_poli(self,superficie):
        posx,posy,separar=20,40,80
        superficie.blit(self.icon_gradomas,(5*posx,posy+separar))
        superficie.blit(self.icon_gradomenos,(5*posx,posy+1.4*separar))
    def parametros_vector(self,superficie):
        posx,posy,separar=20,40,80
        superficie.blit(self.icon_kernel,(5*posx,posy+2*separar))

class LINEAL():
    def __init__(self):
        #MODELO
        x_train,x_test,y_train,y_test=tts(x,y, test_size=0.2)
        lineal=linear_model.LinearRegression()
        lineal.fit(x_train,y_train)
        Y_pred= lineal.predict(x_test)           
        #FORMATO DE IMAGENES EN PYGAME
        self.imagen,self.rect=Formato(x_test,y_test,x_train,y_train,Y_pred)
        #TITULO Y METRICAS        
        metricas=Metricas(y_test,Y_pred)         
        self.texto_grafica=fuentegrande().render("MODELO LINEAL",0,(0,0,0))
        self.texto_metricas=fuentepeque().render("Metricas:"+metricas,0,(0,0,0))
        
    def dibujar(self,superficie):    
        superficie.blit(self.imagen,(400,10)) 
        superficie.blit(self.texto_grafica,(450,10))
        rect_metricas(superficie)        
        superficie.blit(self.texto_metricas,(440,400))

class POLINOMIAL():
    def __init__(self,grado=1):          
        x_train,x_test,y_train,y_test =tts(x,y, test_size=0.2)
        reg_poli= PolynomialFeatures(degree=grado)
        x_entrena_poli= reg_poli.fit_transform(x_train)
        x_test_poli= reg_poli.fit_transform(x_test)
        poli=linear_model.LinearRegression()
        poli.fit(x_entrena_poli,y_train)
        Y_pre_poli=poli.predict(x_test_poli)
        xx = np.arange(min(x_train),max(x_train),0.01)
        coef=[]
        for i in range(0,grado):            
            coef.append(poli.coef_[0][i+1]*xx**(i+1))
        yy=poli.intercept_+ sum(coef)
       
        self.imagen , self.rect=Formatopoli(x_test,y_test,x_train,y_train,Y_pre_poli,xx,yy)   
        metricas=Metricas(y_test,Y_pre_poli)     
        self.texto_grafica=fuentegrande().render("MODELO POLINOMIAL",0,(0,0,0))        
        self.texto_metricas=fuentepeque().render("Metricas:"+metricas,0,(0,0,0))
        
    def dibujar(self,superficie):        
        superficie.blit(self.imagen,(400,10))
        superficie.blit(self.texto_grafica,(450,10))
        rect_metricas(superficie)
        superficie.blit(self.texto_metricas,(440,400))
       
class VECTORES():
    def __init__(self,kernel="linear"):        
        x_train, x_test, y_train, y_test=tts(x,y.ravel(), test_size=0.2)
        vectores= SVR(kernel=kernel,gamma="auto")   
        vectores.fit(x_train,y_train)    
        y_pred_svr= vectores.predict(x_test)
        
        self.imagen , self.rect=Formato(x_test,y_test,x_train,y_train,y_pred_svr)
      
        metricas=Metricas(y_test,y_pred_svr)              
        self.texto_grafica=fuentegrande().render("MODELO VECTORES DE SOPORTE",0,(0,0,0))
        self.texto_metricas=fuentepeque().render("Metricas:"+metricas,0,(0,0,0))
                
    def dibujar(self,superficie):        
        superficie.blit(self.imagen,(400,10))
        superficie.blit(self.texto_grafica,(450,10))
        rect_metricas(superficie)
        superficie.blit(self.texto_metricas,(440,400))

class ARBOL():
    def __init__(self):
        x_train, x_test, y_train, y_test=tts(x,y, test_size=0.2)        
        arbol= DecisionTreeRegressor(max_depth=25)        
        arbol.fit(x_train,y_train)      
        x_grid=np.arange(min(x_test), max(x_test),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        y_pred=arbol.predict(x_grid)
        score=arbol.score(x_train,y_train)     
        
        self.imagen , self.rect=Formato_arboles(x_test,y_test,x_train,y_train,x_grid,y_pred)
        
        self.texto_grafica=fuentegrande().render("MODELO ARBOLES DE DECISION",0,(0,0,0))
        self.texto_metricas=fuentepeque().render("score:"+str(score)[0:6],0,(0,0,0))
                
    def dibujar(self,superficie):        
        superficie.blit(self.imagen,(400,10))
        superficie.blit(self.texto_grafica,(450,10))
        pygame.draw.rect(superficie,pygame.Color("white"),(400,400,460,15),0)
        pygame.draw.rect(superficie,pygame.Color("Black"),(620,400,55,15),1)
        superficie.blit(self.texto_metricas,(580,400))
        
class BOSQUE():
    def __init__(self):
        x_train, x_test, y_train, y_test=tts(x, y, test_size=0.2)       
        bosque= RandomForestRegressor(n_estimators=300,max_depth=100)        
        bosque.fit(x_train,y_train.ravel())      
        y_pred=bosque.predict(x_test)
        x_grid=np.arange(min(x_test), max(x_test),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        y_pred=bosque.predict(x_grid)
        score=bosque.score(x_train,y_train)         
        
        self.imagen , self.rect=Formato_arboles(x_test,y_test,x_train,y_train,x_grid,y_pred)
      
        self.texto_grafica=fuentegrande().render("MODELO BOSQUES ALEATORIOS",0,(0,0,0))
        self.texto_metricas=fuentepeque().render("score:"+str(score)[0:6],0,(0,0,0))
                
    def dibujar(self,superficie):        
        superficie.blit(self.imagen,(400,10))
        superficie.blit(self.texto_grafica,(450,10))
        pygame.draw.rect(superficie,pygame.Color("white"),(400,400,460,15),0)
        pygame.draw.rect(superficie,pygame.Color("Black"),(620,400,55,15),1)
        superficie.blit(self.texto_metricas,(580,400))

class ELASTIC():
    def __init__(self):
        #MODELO
        x_train,x_test,y_train,y_test=tts(x,y, test_size=0.2)
        elastic=linear_model.ElasticNet(alpha=0.01, l1_ratio=1,normalize=True)
        elastic.fit(x_train,y_train)
        Y_pred=elastic.predict(x_test)
        
        #FORMATO DE IMAGENES EN PYGAME
        self.imagen,self.rect=Formato(x_test,y_test,x_train,y_train,Y_pred)
        #TITULO Y METRICAS        
        metricas=Metricas(y_test,Y_pred)         
        self.texto_grafica=fuentegrande().render("MODELO REDES ELASTICAS",0,(0,0,0))
        self.texto_metricas=fuentepeque().render("Metricas:"+metricas,0,(0,0,0))
        
    def dibujar(self,superficie):    
        superficie.blit(self.imagen,(400,10)) 
        superficie.blit(self.texto_grafica,(450,10))
        rect_metricas(superficie)        
        superficie.blit(self.texto_metricas,(440,400))



def Formato(x_test,y_test,x_train,y_train, Y_pred):
    #FORMATO DE IMAGENES DE PYGAME
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_subplot(111)
    ax.plot(x_test,Y_pred,  color="red", label=r"Prediccion")    
    ax.scatter(x_test, y_test, color= "navy",label=r"Datos test")
    ax.scatter(x_train,y_train, color="blue",alpha=0.1,label=r"Datos entrenamiento")
    plt.legend()    
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    imagen_model= pygame.image.fromstring(raw_data, size, "RGB") 
    plt.close(fig)         
    return pygame.image.fromstring(raw_data, size, "RGB"),imagen_model.get_rect()

    
def Formatopoli(x_test,y_test,x_train,y_train, Y_pred,xx,yy):
    #FORMATO DE IMAGENES DE PYGAME
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_subplot(111)    
    ax.plot(xx,yy,  color="red", label=r"Prediccion")    
    ax.scatter(x_test, y_test, color= "navy",label=r"Datos test")
    ax.scatter(x_train,y_train, color="olive",alpha=0.1,label=r"Datos entrenamiento")
    plt.legend()    
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    imagen_model= pygame.image.fromstring(raw_data, size, "RGB") 
    plt.close(fig)         
    return pygame.image.fromstring(raw_data, size, "RGB"),imagen_model.get_rect()

def Formato_arboles(x_test,y_test,x_train,y_train,x_grid,y_pred):
    #FORMATO DE IMAGENES DE PYGAME
    
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_subplot(111)    
    ax.plot(x_grid, y_pred,  color="red", label=r"Prediccion")    
    ax.scatter(x_test, y_test, color= "navy",label=r"Datos test")
    ax.scatter(x_train,y_train, color="yellowgreen",alpha=0.1,label=r"Datos entrenamiento")
    
    plt.legend()    
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    imagen_model= pygame.image.fromstring(raw_data, size, "RGB")
    plt.close(fig)
          
    return pygame.image.fromstring(raw_data, size, "RGB"),imagen_model.get_rect()

def Metricas(y_test,y_pred):
    MAE=metrics.mean_absolute_error(y_test,y_pred)
    MSA=metrics.mean_squared_error(y_test,y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    R2=metrics.r2_score(y_test,y_pred)
    MAE,MSA,RMSE,R2=str(MAE)[0:8],str(MSA)[0:8],str(RMSE)[0:8],str(R2)[0:5]
    metricas="""MAE={}  MSA={}  RMSE={}  R2={}""".format(MAE,MSA,RMSE,R2)
    return metricas
def rect_metricas(superficie):#para que las metricas se muestren bien
        posx,posy=500,400
        blanco=pygame.draw.rect(superficie,pygame.Color("white"),(posx,posy,360,15),0)
        rojo=pygame.draw.rect(superficie,pygame.Color("red"),(800,400,60,15),0)#resaltar R2
        negro=pygame.draw.rect(superficie,pygame.Color("Black"),(posx,posy,360,15),1)
        return blanco, rojo, negro    
def fuentegrande():
    fuente=pygame.font.SysFont("Lucida Sans Typewriter", 23)
    return fuente
def fuentepeque():
    fuente=pygame.font.SysFont("Lucida Sans Typewriter", 12)
    return fuente
def cargar_iconos(posx,posy,ruta):
    icon=pygame.image.load(ruta)
    icon=pygame.transform.scale(icon,(70,70))
    icon_rect=pygame.Rect(posx,posy,70,70)
    return icon, icon_rect
def cargar_iconos_grado(posx,posy,ruta):
    icon=pygame.image.load(ruta)
    icon=pygame.transform.scale(icon,(40,40))
    icon_rect=pygame.Rect(posx,posy,40,40)
    return icon, icon_rect

def Consola():
    pygame.init()
    clock = pygame.time.Clock()
    ventana = pygame.display.set_mode((950, 550))
    pygame.display.set_caption('Modelos de Regresion')
    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    puntero=pygame.Rect(0,0,0,0) #(left,top,base,altura)
    fondo=(240,244,242)
    
#    info=fuentepeque().render("Modelos de regresion",0,(255,0,0))
    grado=1
    kernel_pos=-1
    los_iconos=ICONOS()
    ventana.fill(fondo) 
                      
    run=True
    while run:
       puntero.left, puntero.top=pygame.mouse.get_pos() 
       
           
       for eventos in pygame.event.get():
          if eventos.type == pygame.QUIT:
             run=False
          
          if eventos.type==pygame.MOUSEBUTTONDOWN:
              if puntero.colliderect(los_iconos.rect_lineal):
                  print("modelo lineal") 
                  ventana.fill(fondo)
                  los_iconos.dibujar(ventana)
                  los_iconos.click(LINEAL())                  
              if puntero.colliderect(los_iconos.rect_poli):
                  print("modelo poli")
                  ventana.fill(fondo)
                  los_iconos.click(POLINOMIAL())
                  los_iconos.parametros_poli(ventana)                 
              if puntero.colliderect(los_iconos.rect_vector):
                  print("modelo vectores")
                  ventana.fill(fondo)
                  los_iconos.click(VECTORES())
                  los_iconos.parametros_vector(ventana)  
              if puntero.colliderect(los_iconos.rect_arbol):
                  print("modelo arboles de decision")
                  los_iconos.click(ARBOL())
                  ventana.fill(fondo)
              if puntero.colliderect(los_iconos.rect_bosque):
                  print("modelo bosques aleatorios")
                  los_iconos.click(BOSQUE())
                  ventana.fill(fondo)
              if puntero.colliderect(los_iconos.rect_elastic):
                  print("modelo redes elasticas")
                  los_iconos.click(ELASTIC())
                  ventana.fill(fondo)
              if puntero.colliderect(los_iconos.rect_gradomas):                  
                  grado+=1
                  los_iconos.click(POLINOMIAL(grado))
                  pygame.draw.rect(ventana,fondo,(230,140,40,40),0)
                  texto=fuentegrande().render("Grado:"+str(grado),0,(0,0,0))
                  ventana.blit(texto,(150,140))
                  print("modelo polinomial")
              if puntero.colliderect(los_iconos.rect_gradomenos):                   
                  grado-=1
                  if grado==0: 
                      grado=1
                  los_iconos.click(POLINOMIAL(grado))
                  pygame.draw.rect(ventana,fondo,(230,140,40,40),0)
                  texto=fuentegrande().render("Grado:"+str(grado),0,(0,0,0))
                  ventana.blit(texto,(150,140))
                  print("modelo polinomial")
              if puntero.colliderect(los_iconos.rect_kernel):
                  kernel=["rbf","linear","poly"]
                  kernel_pos+=1
                  if kernel_pos==3:
                      kernel_pos=0                  
                  los_iconos.click(VECTORES(kernel[kernel_pos]))
                  pygame.draw.rect(ventana,pygame.Color("Black"),(115,240,45,15),0)
                  texto=fuentepeque().render(kernel[kernel_pos],0,(250,250,250))
                  ventana.blit(texto,(115,240))
              
          
       los_iconos.dibujar(ventana)

              
       if len(los_iconos.lista_modelos)>0: 
            for x in los_iconos.lista_modelos:
                x.dibujar(ventana)
            

       pygame.display.flip()
       clock.tick(30)
Consola()       
pygame.quit()
