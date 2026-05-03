// spiral_metal_capture.mm
// Programmatic Metal GPU capture, scoped to a brief window.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>

// Capture auto-stops after `duration_seconds` to keep the trace small.
void spiral_metal_capture_start_cpp(const char * path, int duration_seconds) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "[spiral_capture] no Metal device, skipping\n");
            return;
        }

        MTLCaptureManager * mgr = [MTLCaptureManager sharedCaptureManager];

        if (![mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
            fprintf(stderr, "[spiral_capture] GPUTraceDocument not supported. "
                            "Set MTL_CAPTURE_ENABLED=1 in environment.\n");
            return;
        }

        MTLCaptureDescriptor * desc = [[MTLCaptureDescriptor alloc] init];
        desc.captureObject = device;
        desc.destination = MTLCaptureDestinationGPUTraceDocument;

        NSString * nspath = [NSString stringWithUTF8String:path];
        NSURL * url = [NSURL fileURLWithPath:nspath];
        [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
        desc.outputURL = url;

        NSError * err = nil;
        BOOL ok = [mgr startCaptureWithDescriptor:desc error:&err];
        if (!ok) {
            fprintf(stderr, "[spiral_capture] startCapture failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(no error)");
            return;
        }

        fprintf(stderr, "[spiral_capture] capture started -> %s (%d sec window)\n",
                path, duration_seconds);

        std::thread([duration_seconds]() {
            std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
            @autoreleasepool {
                MTLCaptureManager * m = [MTLCaptureManager sharedCaptureManager];
                if ([m isCapturing]) {
                    [m stopCapture];
                    fprintf(stderr, "[spiral_capture] stopped (timer)\n");
                    fflush(stderr);
                }
            }
        }).detach();

        atexit([]() {
            MTLCaptureManager * m = [MTLCaptureManager sharedCaptureManager];
            if ([m isCapturing]) {
                [m stopCapture];
                fprintf(stderr, "[spiral_capture] stopped (atexit)\n");
            }
        });
    }
}
