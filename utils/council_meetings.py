import glob
import os
import textract
from texts.models import Text, Language, TextType, Source

directory = '/vol/tensusers/mbentum/FRYSIAN_ASR/frysian_council_meetings/'
fn = glob.glob(directory + '**/*',recursive = True)
files = [f for f in fn if os.path.isdir(f)]
directories = = [f for f in fn if os.path.isdir(f)]



