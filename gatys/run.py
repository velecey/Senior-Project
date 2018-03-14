from gatys import gatys


""" how to use gatys class,

    ex) 1.   object = gatys(imsize = 128  ## it becomes (128,128)),
                            style_loss_weight = 1000, 
                            content_loss_weight = 1, 
                            epochs = 300)
                            
        2.   object.load_content_img(directory_to_content_img)
             object.load_style_img(directory_to_style_img)
        
        3.   object.choose_laeyrs(content_layer_list = ['conv_4'], 
                                  style_layer_list = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'])
                                  
        4.   object.run_network()
        5.   object.imshow(object.output)


"""

gats = gatys()
gats.load_content_img('Images/yonsei.JPG')
gats.load_style_img('Images/vangogh_starry_night.jpg')
gats.choose_layers()
gats.run_network()
gats.imshow(gats.output)