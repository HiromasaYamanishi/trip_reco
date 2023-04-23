from  PIL import ImageDraw, ImageFont, Image
import os
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
def get_color(coef):
    df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/lime/lime_explain.csv')
    cmap=matplotlib.colors.LinearSegmentedColormap("custom", plt.cm.jet._segmentdata, N=256,gamma=1)
    coef_max = max([max(df[f'coef{i}']) for i in range(10)])
    coef_min= min([min(df[f'coef{i}']) for i in range(10)])
    index = int(255*(coef-coef_min)/(coef_max-coef_min))
    '''
    if coef>0:
        index=int(128*(coef)/(coef_max))+128
    else:
        index = (coef-coef_min)/(-coef_min)*128
    '''
    color=cmap(index)
    color=(int(255*color[0]),int(255*color[1]),int(255*color[2]))
    
    return color

def plot_vocabs(index):
    data_dir = '/home/yamanishi/project/trip_recommend/data/lime'
    font_big = ImageFont.truetype("NotoSerifCJK-Bold.ttc", 40)
    font = ImageFont.truetype("NotoSerifCJK-Bold.ttc", 30)
    font_small = ImageFont.truetype("NotoSerifCJK-Bold.ttc", 20)
    img = Image.fromarray(np.ones((500,500,3),np.uint8)*255)
    # 編集用に画像をコピー
    
    new_img = img.copy()

    # コピーした画像にテキストを追加してplot
    draw = ImageDraw.Draw(new_img)

    df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/lime/lime_explain.csv')
    spot = df.loc[index,'spot_name']
    jalan_image = Image.open(os.path.join('/home/yamanishi/project/trip_recommend/data/jalan_image',f'{spot}_0.jpg'))
    new_img.paste(jalan_image)
    draw.text((170, 20), spot, 'black', font=font_big, anchor='mm')

    gt = df.loc[index, 'gt']
    pred = df.loc[index, 'pred']

    draw.text((50, 70), f'gt: {str(gt.round(3))}','black',font=font, anchor='mm')
    draw.text((300, 70), f'pred: {str(pred.round(3))}','black', font=font, anchor='mm')
    for i in range(10):
        vocab = df.loc[index, f'vocab{i}']
        coef = df.loc[index, f'coef{i}']
        width = [50, 300]
        draw.text((width[i%2], 140+(i//2)*66), vocab,fill=get_color(coef), font=font, anchor='mm', stroke_width=4, stroke_fill='black')
        draw.text((width[i%2], 180+(i//2)*66), str(coef.round(3)),fill=get_color(coef), font=font_small, anchor='mm',stroke_width=4, stroke_fill='black')
    
    plt.imshow(new_img)
    plt.axis('off')
    img_save_dir = os.path.join(data_dir, 'explain_image')
    img_save_path = os.path.join(img_save_dir, f'{spot}.jpg')
    new_img.save(img_save_path)    
plot_vocabs(0)