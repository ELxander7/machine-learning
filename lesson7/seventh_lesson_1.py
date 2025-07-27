# import pygame
# from sklearn.svm import SVC
#
# def main():
#     pygame.init()
#     screen = pygame.display.set_mode((600, 400))
#     screen.fill("white")
#     pygame.display.update()
#
#     X, y = [], []
#     r = 5
#     play = True
#     while(play):
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 play = False
#                 pygame.quit()
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 X.append(event.pos)
#                 if event.button == 1:
#                     pygame.draw.circle(screen, "red", event.pos, r)
#                     y.append("red")
#                 if event.button == 3:
#                     pygame.draw.circle(screen, "blue", event.pos, r)
#                     y.append("blue")
#                 pygame.display.update()
# if __name__ == '__main__':
#     main()
#
# import pygame
# from sklearn.svm import SVC
#
# def main():
#     pygame.init()
#     screen = pygame.display.set_mode((600, 400))
#     screen.fill("white")
#     pygame.display.update()
#
#     X, y = [], []
#     r = 5
#     play = True
#     while(play):
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 play = False
#                 pygame.quit()
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 X.append(event.pos)
#                 if event.button == 1:
#                     pygame.draw.circle(screen, "red", event.pos, r)
#                     y.append("red")
#                 if event.button == 3:
#                     pygame.draw.circle(screen, "blue", event.pos, r)
#                     y.append("blue")
#                 pygame.display.update()
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE:
#                     svc = SVC(kernel="linear")
#                     svc.fit(X, y)
#                     #Ax+By+C=0
#                     A, B = svc.coef_[0]
#                     C = svc.intercept_[0]
#
#                     for i in range(0, 600, 5):
#                         for j in range(0, 400, 5):
#                             pygame.draw.circle(screen, svc.predict([[i,j]])[0], ( i,j),1)
#                             pygame.display.update()
#
#
# if __name__ == '__main__':
#     main()

import pygame
from sklearn.svm import SVC

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    screen.fill("white")
    pygame.display.update()

    X, y = [], []
    r = 5
    play = True
    while(play):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                X.append(event.pos)
                if event.button == 1:
                    pygame.draw.circle(screen, "red", event.pos, r)
                    y.append("red")
                if event.button == 3:
                    pygame.draw.circle(screen, "blue", event.pos, r)
                    y.append("blue")
                pygame.display.update()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if(len(X)>0):
                        svc = SVC(kernel="linear")
                        svc.fit(X, y)
                        # Ax+By+C=0
                        A, B = svc.coef_[0]
                        C = svc.intercept_[0]

                        for i in range(0, 600, 5):
                            for j in range(0, 400, 5):
                                pygame.draw.circle(screen, svc.predict([[i, j]])[0], (i, j), 1)
                                pygame.display.update()


if __name__ == '__main__':
    main()
