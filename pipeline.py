import sys
import math
import gi
gi.require_version('Gst', '1.0')
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer, GstRtsp
import pyds # type: ignore
#from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import configparser


 
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1

MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.4
frame_count = {}
saved_count = {}
perf_data = None


def print_ocr_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        ndfm_pad_index = frame_meta.pad_index
        print(num_rects)
        while l_obj:
            try: 
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                
                break
            l_class = obj_meta.classifier_meta_list

            while l_class is not None:   
                try:
                    class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                except StopIteration:
                    break
                l_label = class_meta.label_info_list
                component_id= class_meta.unique_component_id
                print(component_id)
                
                while l_label is not None:
                    try:
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                    except StopIteration:
                        break
                    ocr_label = label_info.result_label
                    ocr_prob= label_info.result_prob  
                    print(ndfm_pad_index, ocr_label, ocr_prob, component_id)
                    try:
                        l_class=l_class.next
                    except StopIteration:
                        break
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break   
 
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK
 

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index,uri):
    print("Creating source bin")

    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

# Function to create the pipeline
def create_pipeline(args):
    
    #global platform_info
    #platform_info = PlatformInfo()
    
    number_sources=len(args)-1
    
    global perf_data
    perf_data = PERF_DATA(len(args) - 1)

    Gst.init(None)
    
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return None

    elements = {
        "streammux": ("nvstreammux", "Stream-muxer"),
        "nvdspreprocess": ("nvdspreprocess", "preprocess-plugin"),
        "pgie": ("nvinferserver", "primary-inference"),
        "nvvidconv": ("nvvideoconvert", "convertor"),
        "filter1": ("capsfilter", "filter1"),
        #"nvtracker": ("nvtracker", "tracker"),
        #"nvdsanalytics": ("nvdsanalytics", "analytics"),
        "nvtiler": ("nvmultistreamtiler", "nvtiler"),
        "nvvidconv2": ("nvvideoconvert", "convertor2"),
        "nvosd": ("nvdsosd", "onscreendisplay"),
        "nvvidconv_postosd": ("nvvideoconvert", "convertor_postosd"),
        "encoder": ("nvv4l2h264enc", "encoder"),
        "codeparser": ("h264parse", "h264parse"),
        "rtppay": ("rtph264pay", "rtppay"),
        "sink" : ("udpsink", "udpsink")       
    }

    for name, val in elements.items():
        element = Gst.ElementFactory.make(val[0], val[1])
        if not element:
            sys.stderr.write(f"Unable to create {name}\n")
            return None
        elements[name] = element


 

    # config = configparser.ConfigParser()
    # config.read('dsnvanalytics_tracker_config.txt')
    # config.sections()

    # for key in config['tracker']:
    #     if key == 'tracker-width' :
    #         tracker_width = config.getint('tracker', key)
    #         elements["nvtracker"].set_property('tracker-width', tracker_width)
    #     if key == 'tracker-height' :
    #         tracker_height = config.getint('tracker', key)
    #         elements["nvtracker"].set_property('tracker-height', tracker_height)
    #     if key == 'gpu-id' :
    #         tracker_gpu_id = config.getint('tracker', key)
    #         elements["nvtracker"].set_property('gpu_id', tracker_gpu_id)
    #     if key == 'll-lib-file' :
    #         tracker_ll_lib_file = config.get('tracker', key)
    #         elements["nvtracker"].set_property('ll-lib-file', tracker_ll_lib_file)
    #     if key == 'll-config-file' :
    #         tracker_ll_config_file = config.get('tracker', key)
    #         elements["nvtracker"].set_property('ll-config-file', tracker_ll_config_file)
    #elements["nvdsanalytics"].set_property("config-file", "config_nvdsanalytics.txt")



    tiler = elements["nvtiler"]
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    elements["filter1"].set_property("caps", caps1)
    streammux = elements["streammux"]
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    elements["nvdspreprocess"].set_property('config-file', 'config_preprocess.txt')
    elements["pgie"].set_property('config-file-path', "pgie-id-1-ocrnet.txt")
    elements["pgie"].set_property("batch-size", number_sources)

    elements["nvosd"].set_property('process-mode',OSD_PROCESS_MODE)
    elements["nvosd"].set_property('display-text',OSD_DISPLAY_TEXT)

    elements["encoder"].set_property('bitrate', 4097152)
    
    elements["sink"].set_property('host', "127.0.0.1")
    elements["sink"].set_property('port', 8245)
    elements["sink"].set_property('async', False)
    elements["sink"].set_property('sync', 1)
 


    for element in elements.values():
        pipeline.add(element)


    for i in range(number_sources):
        # os.mkdir(folder_name + "/stream_" + str(i))
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i + 1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)


    elements["streammux"].link(elements["nvdspreprocess"])
    elements["nvdspreprocess"].link(elements["pgie"])
    elements["pgie"].link(elements["nvvidconv"])
    #elements["nvdsanalytics"].link(elements["nvvidconv"])
    elements["nvvidconv"].link(elements["filter1"])
    #elements["filter1"].link(elements["nvtracker"])
    #elements["nvtracker"].link(elements["nvdsanalytics"])
    #elements["nvdsanalytics"].link(elements["nvtiler"])
    elements["filter1"].link(elements["nvtiler"])
    elements["nvtiler"].link(elements["nvvidconv2"])
    elements["nvvidconv2"].link(elements["nvosd"])
    elements["nvosd"].link(elements["nvvidconv_postosd"])
    elements["nvvidconv_postosd"].link(elements["encoder"])
    elements["encoder"].link(elements["codeparser"])
    elements["codeparser"].link(elements["rtppay"])
    elements["rtppay"].link(elements["sink"])

    return pipeline, elements["pgie"]

def create_rtsp_server():
    rtsp_port_num = 8554
    rtsp_stream_end = "/live"
    username =  'user'
    password =  "pass"
    updsink_port_num = 8245
    codec = 'H264'

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_protocols(GstRtsp.RTSPLowerTrans.TCP)
    factory.set_transport_mode(GstRtspServer.RTSPTransportMode.PLAY)
    factory.set_latency(1)
    factory.set_launch(
        '( udpsrc name=pay0  port=%d buffer-size=10485760  caps="application/x-rtp, media=video, clock-rate=90000, mtu=1300, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    permissions = GstRtspServer.RTSPPermissions()
    permissions.add_permission_for_role(username, "media.factory.access", True)
    permissions.add_permission_for_role(username, "media.factory.construct", True)
    factory.set_permissions(permissions)
    server.get_mount_points().add_factory(rtsp_stream_end, factory)
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://%s:%s@%s:%d%s ***\n\n" %
        (username, password, 'localhost', rtsp_port_num, rtsp_stream_end))
    
# Function to run the pipeline
def run_pipeline(args):
    print(args)
    create_rtsp_server()
    pipeline, pgie = create_pipeline(args)
    if not pipeline:
        sys.stderr.write("Failed to create pipeline\n")
        return

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # nvanalytics_src_pad=analytics.get_static_pad("src")
    # if not nvanalytics_src_pad:
    #     sys.stderr.write(" Unable to get src pad \n")
    # else:
    #     nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)
    #     # perf callback function to print fps every 5 sec
    #     GLib.timeout_add(5000, perf_data.perf_print_callback)

    pgie_src_pad= pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, print_ocr_src_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except Exception as e:
        raise e
        # pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    run_pipeline(sys.argv)
