import os
import tqdm
import subprocess
import json
from pytube import YouTube, Playlist
from pytube.cli import on_progress
# from moviepy.editor import AudioFileClip

# from requests_html import HTMLSession
# from bs4 import BeautifulSoup as bs
# import re


class YoutubeLoader:

    def __init__(self, args):
        
        self.num_threads: int = args['num_threads']
        self.sr: int = args['sr']

        self.data_dir: str = args['yt_dir']
        self.url_tag: str = "watch?v="
        self.url_pl_tag: str = "playlist?list="

#########################################################################
### ref: https://www.javatpoint.com/how-to-extract-youtube-data-in-python
#########################################################################
    @staticmethod
    def extract_video_informations(url):

        # It will download HTML code
        session = HTMLSession()
        resp = session.get(url)
        # execute Javascript  
        resp.html.render(timeout=60)
        # create beautiful soup object to parse HTML  
        soup = bs(resp.html.html, "html.parser")
        # initialize the result dictionary to store data  
        result = {}

        result["title"] = soup.find("meta", itemprop="name")['content']
        result["views"] = soup.find("meta",itemprop="interactionCount")['content']
        result["description"] = soup.find("meta",itemprop="description")['content']
    
        result["date_published"] = soup.find("meta", itemprop="datePublished")['content']

        result["duration"] = soup.find("span", {"class": "ytp-time-duration"}).text
        result["tags"] = ', '.join([ meta.attrs.get("content") for meta in soup.find_all("meta", {"property": "og:video:tag"}) ])

        data = re.search(r"var ytInitialData = ({.*?});", soup.prettify()).group(1)
        data_json = json.loads(data)
        videoPrimaryInfoRenderer = data_json['contents']['twoColumnWatchNextResults']['results']['results']['contents'][0]['videoPrimaryInfoRenderer']  
        videoSecondaryInfoRenderer = data_json['contents']['twoColumnWatchNextResults']['results']['results']['contents'][1]['videoSecondaryInfoRenderer']  
        # number of likes  
        likes_label = videoPrimaryInfoRenderer['videoActions']['menuRenderer']['topLevelButtons'][0]['segmentedLikeDislikeButtonRenderer']['likeButton']['toggleButtonRenderer']['defaultText']['accessibility']['accessibilityData']['label'] # "No likes" or "###,### likes"  
        # likes_label = videoPrimaryInfoRenderer['videoActions']['menuRenderer']['topLevelButtons'][0]['toggleButtonRenderer']['defaultText']['accessibility']['accessibilityData']['label'] # "No likes" or "###,### likes"  
        likes_str = likes_label.split(' ')[0].replace(',','')  
        result["likes"] = '0' if likes_str == 'No' else likes_str  
    
        channel_tag = soup.find("meta", itemprop="channelId")['content']  
        # channel name  
        channel_name = soup.find("span", itemprop="author").next.next['content']  
        
        channel_url = f"https://www.youtube.com/{channel_tag}"  
        # number of subscribers as str  
        channel_subscribers = videoSecondaryInfoRenderer['owner']['videoOwnerRenderer']['subscriberCountText']['accessibility']['accessibilityData']['label']  
    
        result['channel'] = {'name': channel_name, 'url': channel_url, 'subscribers': channel_subscribers}

        session.close()
        return result

    def save_video_info(self, url, yt_info):
        
        pos = url.find(self.url_tag)+len(self.url_tag)
        yt_name = url[pos:]

        yt_dir = os.path.join(self.data_dir, yt_name)
        os.makedirs(yt_dir, exist_ok=True)

        save_info_path = os.path.join(yt_dir, 'video_info.data')
        with open(save_info_path, 'w') as wf:
            for key, value in yt_info.items():
                if isinstance(value, dict):
                    wf.write("{}\n".format(key))
                    for sub_key, sub_value in value.items():
                        wf.write("\t{}: {}\n".format(sub_key, sub_value))
                elif isinstance(value, list):
                    wf.write("{}\n".format(key))
                    wf.write("\t[")
                    for sub_id, sub_value in enumerate(value):
                        wf.write(sub_value)
                        if len(value) != sub_id-1:
                            wf.write(",")
                    wf.write("]\n")
                else:
                    wf.write("{}: {}\n".format(key, value))


    def dl_high_res_mp4(self, yt, output_path, file_name):

        stream_info = {}

        stream = yt.streams.get_highest_resolution()
        _ = stream.download(output_path=output_path, filename=file_name)

        stream_info['itag'] = stream.itag
        stream_info['mime_type'] = stream.mime_type
        stream_info['resolution'] = stream.resolution
        stream_info['video_fps'] = stream.fps
        stream_info['video_codec'] = stream.video_codec
        stream_info['audio_codec'] = stream.audio_codec
        return stream_info

    def dl_only_audio_mp4(self, yt, output_path, file_name):

        stream_info = {}

        stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').last()
        _ = stream.download(output_path=output_path, filename=file_name)

        stream_info['itag'] = stream.itag
        stream_info['mime_type'] = stream.mime_type
        stream_info['audio_codec'] = stream.audio_codec
        stream_info['abr'] = stream.abr

        return stream_info

    def dl_captions(self, captions, yt_dir, yt_name):

        caption_info = {}
        if len(captions) == 0:
            caption_info["code"]=""
            return caption_info

        yt_xml_dir = os.path.join(yt_dir, "xml")
        os.makedirs(yt_xml_dir, exist_ok=True)

        codes = []
        for caption in captions:
            code = caption.code
            xml_path = os.path.join(yt_xml_dir,"{}_{}.xml".format(yt_name, code))
            out_xml = caption.xml_captions
            with open(xml_path, 'w') as wf:
                wf.write(out_xml)
            codes.append(code)
        caption_info["code"] = ','.join(codes)
        return caption_info

    def convert_mp4_to_wav(self, yt_dir, yt_name):

        hr_video_path = os.path.join(yt_dir, "mp4", yt_name+'.mp4')
        hq_video_path = os.path.join(yt_dir, "mp4", yt_name+'_audio.mp4')
        assert(os.path.exists(hr_video_path) or os.path.exists(hq_video_path)), "Both Youtube Not Downloaded: {}".format(url)

        video_file_path = hq_video_path if os.path.exists(hq_video_path) else hr_video_path

        # save wav
        audio_dir = os.path.join(yt_dir, "wav")
        os.makedirs(audio_dir, exist_ok=True)

        audio_file_path = os.path.join(audio_dir, yt_name+'.wav')

        command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar %d %s -loglevel panic" % \
            (video_file_path, self.num_threads, self.sr, audio_file_path))
        out = subprocess.call(command, shell=True, stdout=None)
        if out != 0:
            raise ValueError("Error: Converting mp4 to wav: {}".format(command))

    def dl_youtube(self, url, dl_caption=True, save_spec_info=False, overwrite=False):

        print("\n>>Download Youtube URL: {}".format(url))

        pos = url.find(self.url_tag)+len(self.url_tag)
        yt_name = url[pos:]

        yt_dir = os.path.join(self.data_dir, yt_name)
        if not overwrite and os.path.exists(yt_dir+"/.done"):
            print(">>Already Exists Youtube: {}".format(url))
            return yt_dir

        yt_mp4_dir = os.path.join(yt_dir, "mp4")

        yt = YouTube(url)

        yt_info = {}
        if save_spec_info:
            yt_info = self.extract_video_informations(url)
        
        if not 'streamingData' in yt.vid_info.keys():
            print(">>Can't Download Youtube: {}".format(url))
            return None

        stream = yt.streams.last()
        yt_info['title'] = stream.title
        yt_info['duration'] = stream._monostate.duration


        yt_info['high_resolution_mp4_info'] = self.dl_high_res_mp4(yt, yt_mp4_dir, yt_name+'.mp4')
        yt_info['only_audio_mp4_info'] = self.dl_only_audio_mp4(yt, yt_mp4_dir, yt_name+'_audio.mp4')
        
        if dl_caption:
            yt_info['captions'] = self.dl_captions(yt.captions, yt_dir, yt_name)

        self.save_video_info(url, yt_info)

        self.convert_mp4_to_wav(yt_dir, yt_name)

        only_audio_mp4_path = os.path.join(yt_mp4_dir, yt_name+'_audio.mp4')
        if os.path.exists(only_audio_mp4_path):
            os.remove(only_audio_mp4_path)

        f = open(yt_dir+"/.done", "w")
        f.close()
        return yt_dir

    def dl_playlist(self, pl_url, overwrite=False):

        p = Playlist(pl_url)
        yt_dir_list = []

        for url in tqdm.tqdm(sorted(p.video_urls)):
            yt_dir = self.dl_youtube(url, overwrite=overwrite)
            if yt_dir is not None:
                yt_dir_list.append(yt_dir)
        return yt_dir_list

    def __call__(self, url, overwrite=False):
        
        print(">>Process Youtube URL: {}".format(url))

        if self.url_tag in url:
            yt_dir = self.dl_youtube(url, overwrite=overwrite)
            if yt_dir is None:
                yt_dir = []
            else:
                yt_dir = [yt_dir]

        elif self.url_pl_tag in url:
            yt_dir = self.dl_playlist(url, overwrite=overwrite)

        else:
            print(">>Youtube url format Error, Skip URL: {}".format(url))

        return yt_dir


if __name__ == '__main__':
    """
    Get an argument parser.
    """
    import argparse
    from whisper.audio import SAMPLE_RATE

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sr', type=int, default=SAMPLE_RATE, required = False, help='sampling rate')
    parser.add_argument('--yt_url', type=str, default='https://www.youtube.com/watch?v=jane6C4rIwc', required=False, help='path of test wav file')
    parser.add_argument('--yt_dir', type=str, default='data/youtube', required=False, help='mp4 download directory')
    parser.add_argument('--num_threads', type=int, default=0, required = False, help='number of threads')

    args = parser.parse_args().__dict__
    url = 'https://www.youtube.com/watch?v=jane6C4rIwc'
    # url = 'https://www.youtube.com/playlist?list=PLrT4uvwaf6uw5ChxpBQnx0dA5fcmXvuB_'
    args['yt_url'] = url

    yt_url: str = args['yt_url']

    yl = YoutubeLoader(args)
    yt_dir_list = yl(yt_url)
