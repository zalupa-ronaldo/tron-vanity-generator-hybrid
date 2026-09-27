#import <Cocoa/Cocoa.h>

#include <filesystem>
#include <signal.h>
#include <string>
#include <vector>

NSString* Redact(NSString* input) {
    NSError* error = nil;
    NSRegularExpression* hex = [NSRegularExpression regularExpressionWithPattern:@"\\b[0-9a-fA-F]{64}\\b"
                                                                             options:0 error:&error];
    NSString* result = [hex stringByReplacingMatchesInString:input options:0
                                                        range:NSMakeRange(0, input.length)
                                                 withTemplate:@"<redacted>"];
    NSRegularExpression* privateField = [NSRegularExpression
        regularExpressionWithPattern:@"(private[_ -]?key\\s*[:=]\\s*[\\\"']?)[0-9a-fA-F]{8,}"
                                 options:NSRegularExpressionCaseInsensitive error:&error];
    return [privateField stringByReplacingMatchesInString:result options:0
                                                     range:NSMakeRange(0, result.length)
                                              withTemplate:@"$1<redacted>"];
}

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, strong) NSWindow* window;
@property(nonatomic, strong) NSPopUpButton* backend;
@property(nonatomic, strong) NSTextField* dictionary;
@property(nonatomic, strong) NSTextField* results;
@property(nonatomic, strong) NSTextField* seconds;
@property(nonatomic, strong) NSButton* resident;
@property(nonatomic, strong) NSButton* unique;
@property(nonatomic, strong) NSButton* start;
@property(nonatomic, strong) NSButton* stop;
@property(nonatomic, strong) NSButton* selfTest;
@property(nonatomic, strong) NSButton* benchmark;
@property(nonatomic, strong) NSButton* openResults;
@property(nonatomic, strong) NSTextField* status;
@property(nonatomic, strong) NSProgressIndicator* progress;
@property(nonatomic, strong) NSTextView* log;
@property(nonatomic, strong) NSTask* task;
@property(nonatomic, strong) NSPipe* pipe;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    (void)notification;
    [self buildWindow];
}

- (void)buildWindow {
    NSRect frame = NSMakeRect(0, 0, 960, 680);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                               styleMask:(NSWindowStyleMaskTitled |
                                                          NSWindowStyleMaskClosable |
                                                          NSWindowStyleMaskResizable |
                                                          NSWindowStyleMaskMiniaturizable)
                                                 backing:NSBackingStoreBuffered defer:NO];
    self.window.title = @"TRON Vanity Generator";
    self.window.delegate = self;
    [self.window center];

    NSView* content = self.window.contentView;
    CGFloat width = frame.size.width;
    CGFloat y = frame.size.height - 42;

    [self addLabel:@"Backend" frame:NSMakeRect(16, y, 105, 24) to:content];
    self.backend = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(126, y - 2, 150, 28) pullsDown:NO];
    [self.backend addItemsWithTitles:@[@"auto", @"opencl", @"vulkan", @"cuda", @"metal", @"cpu"]];
    [content addSubview:self.backend];

    [self addLabel:@"Seconds (0 = Stop)" frame:NSMakeRect(300, y, 140, 24) to:content];
    self.seconds = [self field:@"0" frame:NSMakeRect(442, y - 2, 70, 28)];
    [content addSubview:self.seconds];
    self.resident = [self checkbox:@"Resident GPU (OpenCL/Metal)" frame:NSMakeRect(530, y - 2, 210, 28)];
    [content addSubview:self.resident];
    self.unique = [self checkbox:@"First address per word" frame:NSMakeRect(746, y - 2, 190, 28)];
    [content addSubview:self.unique];

    y -= 39;
    [self addLabel:@"Dictionary" frame:NSMakeRect(16, y, 105, 24) to:content];
    self.dictionary = [self field:@"" frame:NSMakeRect(126, y - 2, width - 235, 28)];
    [content addSubview:self.dictionary];
    [content addSubview:[self button:@"Browse..." action:@selector(browseDictionary:) frame:NSMakeRect(width - 96, y - 2, 80, 28)]];

    y -= 39;
    [self addLabel:@"Results folder" frame:NSMakeRect(16, y, 105, 24) to:content];
    self.results = [self field:@"" frame:NSMakeRect(126, y - 2, width - 235, 28)];
    [content addSubview:self.results];
    [content addSubview:[self button:@"Browse..." action:@selector(browseResults:) frame:NSMakeRect(width - 96, y - 2, 80, 28)]];

    y -= 43;
    self.start = [self button:@"Start search" action:@selector(startSearch:) frame:NSMakeRect(126, y, 120, 30)];
    self.stop = [self button:@"Stop" action:@selector(stopProcess:) frame:NSMakeRect(254, y, 85, 30)];
    self.selfTest = [self button:@"GPU self-test" action:@selector(startSelfTest:) frame:NSMakeRect(347, y, 120, 30)];
    self.benchmark = [self button:@"Resident benchmark" action:@selector(startBenchmark:) frame:NSMakeRect(475, y, 145, 30)];
    self.openResults = [self button:@"Open results" action:@selector(openResults:) frame:NSMakeRect(628, y, 120, 30)];
    [content addSubview:self.start];
    [content addSubview:self.stop];
    [content addSubview:self.selfTest];
    [content addSubview:self.benchmark];
    [content addSubview:self.openResults];

    y -= 31;
    self.status = [self field:@"Ready" frame:NSMakeRect(16, y, width - 220, 24)];
    self.status.bezeled = NO;
    self.status.editable = NO;
    self.status.drawsBackground = NO;
    [content addSubview:self.status];
    self.progress = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(width - 195, y + 3, 180, 16)];
    self.progress.indeterminate = NO;
    self.progress.minValue = 0;
    self.progress.maxValue = 100;
    [content addSubview:self.progress];

    NSScrollView* scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(16, 16, width - 32, y - 26)];
    scroll.hasVerticalScroller = YES;
    scroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.log = [[NSTextView alloc] initWithFrame:scroll.bounds];
    self.log.editable = NO;
    self.log.font = [NSFont userFixedPitchFontOfSize:12];
    self.log.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    scroll.documentView = self.log;
    [content addSubview:scroll];

    self.stop.enabled = NO;
    [self setDefaultPaths];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)addLabel:(NSString*)text frame:(NSRect)frame to:(NSView*)view {
    NSTextField* label = [self field:text frame:frame];
    label.bezeled = NO;
    label.editable = NO;
    label.drawsBackground = NO;
    [view addSubview:label];
}

- (NSTextField*)field:(NSString*)value frame:(NSRect)frame {
    NSTextField* field = [[NSTextField alloc] initWithFrame:frame];
    field.stringValue = value;
    return field;
}

- (NSButton*)checkbox:(NSString*)title frame:(NSRect)frame {
    NSButton* button = [[NSButton alloc] initWithFrame:frame];
    button.title = title;
    button.buttonType = NSButtonTypeSwitch;
    return button;
}

- (NSButton*)button:(NSString*)title action:(SEL)action frame:(NSRect)frame {
    NSButton* button = [[NSButton alloc] initWithFrame:frame];
    button.title = title;
    button.bezelStyle = NSBezelStyleRounded;
    button.target = self;
    button.action = action;
    return button;
}

- (NSString*)executableDirectory {
    return [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
}

- (NSString*)generatorPath {
    return [[self executableDirectory] stringByAppendingPathComponent:@"tron_vanity_generator"];
}

- (void)setDefaultPaths {
    NSString* directory = [self executableDirectory];
    NSArray<NSString*>* candidates = @[
        [directory stringByAppendingPathComponent:@"words.txt"],
        [directory stringByAppendingPathComponent:@"words.long.8plus.txt"],
        [[[NSFileManager defaultManager] currentDirectoryPath]
            stringByAppendingPathComponent:@"words.txt"]
    ];
    for (NSString* candidate in candidates) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
            self.dictionary.stringValue = candidate;
            break;
        }
    }
    if (!self.dictionary.stringValue.length)
        self.dictionary.stringValue = [directory stringByAppendingPathComponent:@"words.txt"];
    self.results.stringValue = [directory stringByAppendingPathComponent:@"results"];
}

- (NSString*)backendName {
    return self.backend.selectedItem.title ?: @"auto";
}

- (BOOL)baseArguments:(NSMutableArray<NSString*>*)arguments {
    if (![[NSFileManager defaultManager] fileExistsAtPath:[self generatorPath]]) {
        [self alert:@"tron_vanity_generator was not found next to the GUI.\nCopy both files from the release archive into one folder." title:@"Generator not found"];
        return NO;
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:self.dictionary.stringValue]) {
        [self alert:@"Choose an existing dictionary file first." title:@"Dictionary"];
        return NO;
    }
    if (!self.results.stringValue.length) {
        [self alert:@"Choose a results folder first." title:@"Results"];
        return NO;
    }
    [arguments addObjectsFromArray:@[@"--no-config", @"--backend", [self backendName], @"--words", self.dictionary.stringValue,
                                     @"--out", self.results.stringValue]];
    return YES;
}

- (void)startSearch:(id)sender {
    (void)sender;
    if (self.task) return;
    NSMutableArray<NSString*>* args = [NSMutableArray array];
    if (![self baseArguments:args]) return;
    NSString* seconds = self.seconds.stringValue.length ? self.seconds.stringValue : @"0";
    [args addObjectsFromArray:@[@"--seconds", seconds]];
    NSString* backend = [self backendName];
    if (self.resident.state == NSControlStateValueOn && ![backend isEqualToString:@"vulkan"] &&
        ![backend isEqualToString:@"cpu"])
        [args addObject:@"--gpu-resident"];
    if (self.unique.state == NSControlStateValueOn) [args addObject:@"--unique-words"];
    [self startTask:args label:@"Searching"];
}

- (void)startSelfTest:(id)sender {
    (void)sender;
    if (self.task) return;
    NSMutableArray<NSString*>* args = [NSMutableArray arrayWithObjects:@"--no-config", @"--backend", [self backendName], @"--gputest", nil];
    if (self.resident.state == NSControlStateValueOn && [[self backendName] isEqualToString:@"opencl"])
        [args addObject:@"--gpu-resident"];
    [self startTask:args label:@"GPU self-test"];
}

- (void)startBenchmark:(id)sender {
    (void)sender;
    if (self.task) return;
    NSMutableArray<NSString*>* args = [NSMutableArray array];
    if (![self baseArguments:args]) return;
    [args addObjectsFromArray:@[@"--bench-resident", @"--bench-seconds", @"5"]];
    [self startTask:args label:@"Resident benchmark"];
}

- (void)startTask:(NSArray<NSString*>*)arguments label:(NSString*)label {
    self.pipe = [NSPipe pipe];
    self.task = [NSTask new];
    self.task.launchPath = [self generatorPath];
    self.task.arguments = arguments;
    self.task.currentDirectoryURL = [NSURL fileURLWithPath:[self executableDirectory]];
    self.task.standardOutput = self.pipe;
    self.task.standardError = self.pipe;
    NSFileHandle* reader = self.pipe.fileHandleForReading;
    __weak AppDelegate* weakSelf = self;
    reader.readabilityHandler = ^(NSFileHandle* handle) {
        NSData* data = [handle availableData];
        if (!data.length) {
            handle.readabilityHandler = nil;
            return;
        }
        NSString* text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
        if (!text) text = [[NSString alloc] initWithData:data encoding:NSASCIIStringEncoding];
        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate* strongSelf = weakSelf;
            if (strongSelf && text) [strongSelf appendLog:Redact(text)];
        });
    };
    self.task.terminationHandler = ^(NSTask* task) {
        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate* strongSelf = weakSelf;
            if (strongSelf) [strongSelf taskFinished:task.terminationStatus label:label];
        });
    };
    [self appendLog:[NSString stringWithFormat:@"\n--- %@ ---\n", label]];
    self.status.stringValue = [label stringByAppendingString:@" running..."];
    self.start.enabled = NO;
    self.selfTest.enabled = NO;
    self.benchmark.enabled = NO;
    self.stop.enabled = YES;
    @try {
        [self.task launch];
    } @catch (NSException* exception) {
        self.task = nil;
        self.stop.enabled = NO;
        self.start.enabled = YES;
        self.selfTest.enabled = YES;
        self.benchmark.enabled = YES;
        [self alert:exception.reason ?: @"Could not start generator" title:@"Launch error"];
    }
}

- (void)stopProcess:(id)sender {
    (void)sender;
    if (!self.task) return;
    self.status.stringValue = @"Stopping...";
    [self.task interrupt];
    NSTask* task = self.task;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        if (task == self.task && task.isRunning) [task terminate];
    });
}

- (void)taskFinished:(int)status label:(NSString*)label {
    if (!self.task) return;
    self.task = nil;
    self.stop.enabled = NO;
    self.start.enabled = YES;
    self.selfTest.enabled = YES;
    self.benchmark.enabled = YES;
    self.status.stringValue = status == 0 ? @"Finished successfully" :
        (status == SIGINT ? @"Stopped" : @"Finished with an error");
    if (status != 0) [self appendLog:[NSString stringWithFormat:@"\n%@ exit code: %d\n", label, status]];
}

- (void)appendLog:(NSString*)text {
    if (!text.length) return;
    NSString* safe = Redact(text);
    [[self.log textStorage] appendAttributedString:[[NSAttributedString alloc] initWithString:safe]];
    [self.log scrollRangeToVisible:NSMakeRange(self.log.string.length, 0)];
    NSArray<NSString*>* lines = [safe componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSString* line = lines.lastObject;
    if (!line.length && lines.count > 1) line = lines[lines.count - 2];
    if (line.length && line.length < 220) self.status.stringValue = line;
    NSRange run = [line rangeOfString:@"| run "];
    if (run.location != NSNotFound) {
        NSString* value = [line substringFromIndex:run.location + run.length];
        NSScanner* scanner = [NSScanner scannerWithString:value];
        double percent = 0;
        if ([scanner scanDouble:&percent] && percent >= 0 && percent <= 100) self.progress.doubleValue = percent;
    }
}

- (void)browseDictionary:(id)sender {
    (void)sender;
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    if ([panel runModal] == NSModalResponseOK) self.dictionary.stringValue = panel.URL.path;
}

- (void)browseResults:(id)sender {
    (void)sender;
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = NO;
    panel.canChooseDirectories = YES;
    panel.canCreateDirectories = YES;
    panel.allowsMultipleSelection = NO;
    if ([panel runModal] == NSModalResponseOK) self.results.stringValue = panel.URL.path;
}

- (void)openResults:(id)sender {
    (void)sender;
    [[NSFileManager defaultManager] createDirectoryAtPath:self.results.stringValue
                              withIntermediateDirectories:YES attributes:nil error:nil];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:self.results.stringValue]];
}

- (void)alert:(NSString*)message title:(NSString*)title {
    NSAlert* alert = [NSAlert new];
    alert.messageText = title;
    alert.informativeText = message;
    [alert runModal];
}

- (void)applicationWillTerminate:(NSNotification*)notification {
    (void)notification;
    if (self.task && self.task.isRunning) [self.task terminate];
}

@end

int main(int argc, const char* argv[]) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        NSApplication* application = [NSApplication sharedApplication];
        AppDelegate* delegate = [AppDelegate new];
        application.delegate = delegate;
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
        [application run];
    }
    return 0;
}
