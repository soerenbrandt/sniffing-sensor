import time
from twilio.rest import TwilioRestClient


# # Twilio phone number goes here. Grab one at https://twilio.com/try-twilio
# # and use the E.164 format, for example: "+12025551234"
# TWILIO_PHONE_NUMBER = "+12562697167"

# # list of one or more phone numbers to dial, in "+19732644210" format
# DIAL_NUMBERS = ["+18573205840"]

# # URL location of TwiML instructions for how to handle the phone call
# TWIML_INSTRUCTIONS_URL = \
#   "http://static.fullstackpython.com/phone-calls-python.xml"

# # replace the placeholder values with your Account SID and Auth Token
# # found on the Twilio Console: https://www.twilio.com/console
# client = TwilioRestClient("ACafcdedd80029cdb4a1da432f89eb90a0", "dfd15f64d5b420731ad1d011e64a3325")


# def dial_numbers(numbers_list):
#     """Dials one or more phone numbers from a Twilio phone number."""
#     for number in numbers_list:
#         print("Dialing " + number)
#         # set the method to "GET" from default POST because Amazon S3 only
#         # serves GET requests on files. Typically POST would be used for apps
#         client.calls.create(to=number, from_=TWILIO_PHONE_NUMBER,
#                             url=TWIML_INSTRUCTIONS_URL, method="GET")

# Odorant to time
def csv_to_time(odorant_time):
    try: 
        csv_list = list(" ".join(odorant_time.strip().split())) + [" "]
        numlist = []
        num = []

        for i in csv_list:
            isnum = i.isnumeric()
            if num == [] and isnum: 
                num.append(i)
            elif isnum:
                num.append(i)
            elif num != [] and not isnum: 
                numlist.append(''.join(map(str,num)))
                num = []

        return [float(i) for i in numlist]
    except:
        print("Invalid input")
        

# River plot

def plot_river(exp, plt, np):
    # Plot river plot
    fig, ax = plt.subplots()
    #fig = plt.figure(frameon=False)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #fig.add_axes(ax)
    #fig.set_size_inches(5,5)

    # Plot figure (adapted from: fra_plotting.image_plot_2d(std_data,subtle=True, cmap = 'jet') )
    if type(exp) == np.ndarray:
        d_array = exp
        extent = [0, 1, 1, 0]
    else:
        d_array = exp.spectra
        extent = [np.min(exp.wavelengths), np.max(exp.wavelengths), np.max(exp.times), np.min(exp.times)]

    vmin = np.amin(d_array)
    vmax = np.amax(d_array)
    plt.imshow(d_array, vmin=vmin, vmax=vmax, interpolation='none', cmap='jet')
    
    # Define image labels
    plt.ylabel('Time (s)', fontsize=16)
    plt.xlabel('Wavelength (nm)', fontsize=16)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x',colors='black')
    ax.tick_params(axis='y',colors='black')
    
    # Image aspect ratio
    xext, yext = plt.gca().axes.get_xlim(), plt.gca().axes.get_ylim()
    xrange = xext[1] - xext[0]
    yrange = yext[1] - yext[0]
    plt.gca().set_aspect(1 * abs(xrange / yrange)) # This is the line that causes the warnings about unicode
    
    plt.tight_layout()
    
    
    
# Sleep
def Sleep(sleep_time, ser, intensities, spec, delay_time, elapse, last=False):  
    last_time = elapse[-1]
    i = 0
    start_time = time.time()

    while i < sleep_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        
        if last: 
            if round(sleep_time-i, 1) in [j for j in range(3)]: 
                print("Sleep ends in: {}".format(round(sleep_time-i)), end="\r")
            
        time.sleep(1/delay_time)

# Clean time
def Purge(purge_time, ser): 
    ser.write(b'AB')
    time.sleep(purge_time)
    ser.write(b'ab')
    
def Nitrogen(clean_time, ser, intensities, spec, elapse, purge=False):
    ser.write(b'A')
    
    last_time = elapse[-1]
    i = 0
    start_time = time.time()

    while i < clean_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1)
        
        if round(clean_time-i) in [i for i in range(10)]: 
            print(round(clean_time-i))
    
    ser.write(b'a')
    
# Scan time

def valveD(scan_time, ser, intensities, spec, delay_time, elapse):
    ser.write(b'D')
    last_time = elapse[-1]
    i = 0
    start_time = time.time()
    
    while i < scan_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1/delay_time)
        
    ser.write(b'd')
    
# Water

def valveC(scan_time, ser, intensities, spec, delay_time, elapse):
    
    ser.write(b'C')
    last_time = elapse[-1]
    i = 0
    start_time = time.time()
    
    while i < scan_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1/delay_time)
        
    ser.write(b'c')
    

def valveB(scan_time, ser, intensities, spec, delay_time, elapse):
    ser.write(b'B')
    last_time = elapse[-1]
    i = 0
    start_time = time.time()
    
    while i < scan_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1/delay_time)
        
    ser.write(b'b')
    