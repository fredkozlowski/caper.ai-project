# Caper.ai Project Write-Up

Hi Caper.ai Team Member! I wanted to create a mini-project to show my interest in joining Caper.ai. I built a tiny OpenCV program that tracks when an item is inserted into a cart. Doing this gave me a feeling for what technical challenges making the Caper Cart might involve.

My code is in the cart.py file. I used Python 3 and OpenCV 4.5.3. Please let me know if you hit any issues running it.

A general overview of how my code works is given below -

I used pretrained YOLO object detection which was trained on the COCO dataset. It then detects a human based on webcam images. I chose a bottle as a stand in for a generic grocery item, since it's easier for you to test on your computer. If an object is detected, but then loses detection for 4 frames, the test for whether an object was inserted into the cart is applied. If the last known vertical location is lower than the median vertical location of that object, it's assumed that the object is inserted.

I've also added barcode detection to the code. My thinking behind this was that the chances of detecting a barcode was fairly low, but highly accurate. If a barcode is detected, that's a very good identifier for the product being added to the cart. My code currently detects the barcode, then applies the perspective transform so that the barcode appears clearly. It would be trivial to then apply a barcode reading library, however my Macbook was unable to install it. 

I noticed a few potential roadblocks for the Caper Cart. I'd be very interested to learn how the company has worked to resolve these

- Bulk items like multiple paper towel rolls. I noticed the Caper Cart doesn't have the usual rack underneath like a regular shopping cart does.

- Localization in the store. I assume you are using Bluetooth beacons since I'm not sure if there's any good alternative. SLAM is too computationally expensive and GPS isn't accurate enough. 

- Produce bags. How would the cameras detect a fruit or vegetable in a plastic bag, obscuring the object? I'm wondering if the current solution is manually keying in the item.

- Tracking whether an object has entered or left the cart. My first instinct was to use linear regression on the vertical position of the object vs time data, but this isn't robust against indecisive item insertion. What I chose was a simpler and more abstract method. Looking at the object's last known vertical position and checking whether that's below the median value of the object's vertical positions seems to give better results. 

- I didn't add a weight sensor, but I'm curious if you've resolved the "unexpected item in bagging area" error. While it's probably correctly flagged as an error from a technical standpoint, the UX of encountering that is terrible.

TESTING INSTRUCTIONS

$ python3 cart.py

In case it doesn't run, I have also included the output from a test run of the code. Please note that due to age, my computer ran this very slowly, which is why my sample output is choppy.

Thank you for reading!
