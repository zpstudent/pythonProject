# -*- coding: utf-8 -*-
# @File:moon cake.py
# @Author:south wind
# @Date:2022-09-04
# @IDE:PyCharm
import turtle

t = turtle.Pen()  # 画笔一 用于画图
t.speed(0)


# 花纹颜色 #F29407
# 饼身颜色 #F8B41A

# 画 饼身部分
def outfill_flower(flower_num: "花瓣数量", flower_color: "花瓣颜色"):
    for i in range(flower_num):
        t.left(i * (360 // flower_num))
        t.color(flower_color)
        t.penup()
        t.forward(200)
        t.pendown()
        t.fillcolor(flower_color)
        t.begin_fill()
        t.circle(60)
        t.end_fill()
        t.penup()
        t.home()


# 画 饼身外围 花纹部分
def out_line_flower(flower_num: "花纹数量", flower_color: "花纹颜色"):
    for i in range(flower_num):
        t.pensize(5)
        t.left(i * (360 // 18))
        t.color(flower_color)
        t.penup()
        t.forward(192)
        t.pendown()
        t.circle(60)
        t.penup()
        t.home()


# 画内测的大圆 大圆的填充色比饼身略亮
def big_circle(circle_color: "大圆颜色", circle_fill_color: "大圆填充颜色", circle_size: "大圆半径"):
    t.goto(circle_size, 0)
    t.left(90)
    t.pendown()
    t.pensize(8)
    t.color(circle_color)
    t.fillcolor(circle_fill_color)
    t.begin_fill()
    t.circle(circle_size)
    t.end_fill()
    t.penup()
    t.home()


# 饼上印花文字 文字内容和坐标用字典存储
def write_font(text_content: "文本内容", text_color: "文本颜色", size: "文字大小"):
    t.color(text_color)
    for x in text_content:
        t.penup()
        t.goto(text_content[x])
        t.write(x, align='center', font=('simhei', size, 'bold'))
    t.penup()
    t.home()
    t.color('#F29407')


# 饼身中间矩形条纹部分
def body_center_line(width: "矩形宽度", height: "矩形高度"):
    t.penup()
    t.home()
    t.pensize(4)
    t.pendown()
    t.backward(width / 2)
    t.forward(width)
    t.left(90)
    t.forward(height)
    t.left(90)
    t.forward(width)
    t.left(90)
    t.forward(height * 2)
    t.left(90)
    t.forward(width)
    t.left(90)
    t.forward(height)
    t.penup()
    t.home()


# 矩形条纹两侧的四个花纹 画笔轨迹是一样的 所以只需要传入不同的初始位置和角度即可复用代码
def center_flower(start_point: "落笔位置", start_angle: "落笔朝向", angle_direction_change: "新朝向",
                  rectangle_height: "矩形高度", circle_direction: "花纹弧度"):
    t.penup()
    t.goto(start_point)
    t.pendown()
    t.setheading(start_angle)
    t.forward(10)
    t.setheading(angle_direction_change)
    t.forward(20)
    t.backward(rectangle_height * 2)
    t.forward(rectangle_height * 2)
    t.setheading(start_angle)
    t.circle(circle_direction * 70, 90)
    t.setheading(start_angle + 180)
    t.forward(60)
    t.setheading(angle_direction_change)
    t.forward(30)
    t.penup()
    t.home()


# 饼身上下左右的花纹
def out_flower(begin_x: "落笔横坐标", begin_y: "落笔纵坐标", start_angle: "落笔朝向"):
    t.penup()
    t.goto(begin_x, begin_y)
    t.pendown()
    t.setheading(start_angle)
    t.forward(20)
    t.right(90)
    t.circle(-100, 20)

    t.penup()
    t.goto(begin_x, begin_y)
    t.pendown()
    t.setheading(start_angle)
    t.right(90)
    t.circle(-100, 30)
    t.left(90)
    t.forward(45)
    t.left(95)
    t.circle(190, 50)
    t.left(95)
    t.forward(45)
    t.left(90)
    t.circle(-100, 31)
    t.setheading(start_angle)
    t.forward(20)
    t.left(90)
    t.circle(100, 20)
    t.penup()
    t.home()


# 以下代码开始调用各种功能
if __name__ == "__main__":
    # 设置画布名称
    t.screen.title('中秋快乐')
    # 画 饼身部分
    outfill_flower(18, '#F8B41A')
    # 画 饼身外围 花纹部分
    out_line_flower(18, '#F29407')
    # 画内测的大圆 大圆的填充色比饼身略亮
    # big_circle('#F29407','#F8B41A',200)
    big_circle('#F29407', '#F8B51D', 200)
    # 饼上印花文字 文字内容和坐标用字典存储
    text_content = {'中': (-100, 70), '秋': (100, 70), '乐': (100, -120), '快': (-98, -125)}  # 圆字坐标最后向下微调了一下
    # write_font(text_content,'#F29407',40)
    write_font(text_content, '#FC932B', 40)
    # 饼身中间矩形条纹部分
    body_center_line(12, 80)
    # 矩形条纹两侧的四个花纹
    center_flower((6, 60), 0, 90, 80, -1)
    center_flower((6, -60), 0, -90, 80, 1)
    center_flower((-6, 60), 180, 90, 80, 1)
    center_flower((-6, -60), 180, -90, 80, -1)
    # 饼身上下左右的花纹
    out_flower(6, 110, 90)
    out_flower(-110, 6, 180)
    out_flower(-6, -110, 270)
    out_flower(110, -6, 360)
    # 可以再加点字
    # text_content2 = {'天': (-50, 30), '地': (50, 30), '仁': (50, -60), '和': (-50, -60)}  # 圆字坐标最后向下微调了一下
    # write_font(text_content2, '#F29407',30)

    # 隐藏画笔
    t.hideturtle()
    # 保持画布显示
    turtle.done()





