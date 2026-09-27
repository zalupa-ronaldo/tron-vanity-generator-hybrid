#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iterator>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr UINT WM_GUI_OUTPUT = WM_APP + 1;
constexpr UINT WM_GUI_EXIT = WM_APP + 2;

enum ControlId : int {
    IdBackend = 100,
    IdDictionary = 101,
    IdBrowseDictionary = 102,
    IdResults = 103,
    IdBrowseResults = 104,
    IdSeconds = 105,
    IdResident = 106,
    IdUnique = 107,
    IdStart = 108,
    IdStop = 109,
    IdSelfTest = 110,
    IdBenchmark = 111,
    IdOpenResults = 112,
    IdLog = 113,
    IdStatus = 114,
    IdProgress = 115,
};

std::wstring quoteArg(const std::wstring& value) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t ch : value) {
        if (ch == L'\\') {
            ++slashes;
        } else if (ch == L'\"') {
            out.append(slashes * 2 + 1, L'\\');
            out.push_back(ch);
            slashes = 0;
        } else {
            out.append(slashes, L'\\');
            slashes = 0;
            out.push_back(ch);
        }
    }
    out.append(slashes * 2, L'\\');
    out.push_back(L'\"');
    return out;
}

std::wstring fromUtf8(const std::string& value) {
    if (value.empty()) return {};
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                    value.data(), static_cast<int>(value.size()), nullptr, 0);
    UINT codePage = CP_UTF8;
    if (!count) {
        codePage = CP_ACP;
        count = MultiByteToWideChar(codePage, 0, value.data(), static_cast<int>(value.size()),
                                    nullptr, 0);
    }
    std::wstring result(static_cast<size_t>(count), L'\0');
    if (count) MultiByteToWideChar(codePage, 0, value.data(), static_cast<int>(value.size()),
                                   result.data(), count);
    return result;
}

std::wstring redact(const std::wstring& text) {
    // The generator normally prints only public addresses, but keep the GUI
    // safe if a future diagnostic accidentally emits a 32-byte hex secret.
    static const std::wregex hexKey(LR"(\b[0-9a-fA-F]{64}\b)");
    static const std::wregex privateField(
        LR"((private[_ -]?key\s*[:=]\s*["']?)[0-9a-fA-F]{8,})",
        std::regex_constants::icase);
    std::wstring result = std::regex_replace(text, hexKey, L"<redacted>");
    return std::regex_replace(result, privateField, L"$1<redacted>");
}

class GuiApp {
public:
    int run(HINSTANCE instance) {
        instance_ = instance;
        INITCOMMONCONTROLSEX common{sizeof(common), ICC_PROGRESS_CLASS};
        InitCommonControlsEx(&common);

        WNDCLASSW wc{};
        wc.hInstance = instance_;
        wc.lpfnWndProc = &GuiApp::windowProc;
        wc.lpszClassName = L"TronVanityGuiWindow";
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
        if (!RegisterClassW(&wc)) return 1;

        hwnd_ = CreateWindowExW(0, wc.lpszClassName, L"TRON Vanity Generator",
                                WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                                CW_USEDEFAULT, CW_USEDEFAULT, 920, 680,
                                nullptr, nullptr, instance_, this);
        if (!hwnd_) return 1;
        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);

        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0) > 0) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        return static_cast<int>(message.wParam);
    }

private:
    static LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
        GuiApp* app = reinterpret_cast<GuiApp*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (message == WM_NCCREATE) {
            const auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
            app = static_cast<GuiApp*>(create->lpCreateParams);
            app->hwnd_ = hwnd;
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
        }
        return app ? app->handleMessage(message, wParam, lParam)
                   : DefWindowProcW(hwnd, message, wParam, lParam);
    }

    LRESULT handleMessage(UINT message, WPARAM wParam, LPARAM lParam) {
        switch (message) {
        case WM_CREATE:
            createControls();
            return 0;
        case WM_SIZE:
            layoutControls(LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_COMMAND:
            handleCommand(LOWORD(wParam));
            return 0;
        case WM_GUI_OUTPUT: {
            auto* text = reinterpret_cast<std::wstring*>(lParam);
            if (text) {
                appendLog(*text);
                delete text;
            }
            return 0;
        }
        case WM_GUI_EXIT: {
            auto* code = reinterpret_cast<DWORD*>(lParam);
            const DWORD exitCode = code ? *code : 1;
            delete code;
            finishProcess(exitCode);
            return 0;
        }
        case WM_CLOSE:
            shutdownProcess();
            DestroyWindow(hwnd_);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd_, message, wParam, lParam);
        }
    }

    static HWND makeControl(HWND parent, const wchar_t* className, const wchar_t* text,
                            DWORD style, int id) {
        return CreateWindowExW(0, className, text, WS_CHILD | WS_VISIBLE | style,
                               0, 0, 0, 0, parent, reinterpret_cast<HMENU>(id),
                               GetModuleHandleW(nullptr), nullptr);
    }

    void createControls() {
        font_ = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
        labelBackend_ = makeControl(hwnd_, L"STATIC", L"Backend", SS_LEFT, -1);
        backend_ = makeControl(hwnd_, WC_COMBOBOXW, L"", CBS_DROPDOWNLIST | WS_TABSTOP, IdBackend);
        for (const wchar_t* name : {L"auto", L"opencl", L"vulkan", L"cuda", L"metal", L"cpu"})
            SendMessageW(backend_, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(name));
        SendMessageW(backend_, CB_SETCURSEL, 0, 0);

        labelDictionary_ = makeControl(hwnd_, L"STATIC", L"Dictionary", SS_LEFT, -1);
        dictionary_ = makeControl(hwnd_, L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP,
                                  IdDictionary);
        browseDictionary_ = makeControl(hwnd_, L"BUTTON", L"Browse...", BS_PUSHBUTTON | WS_TABSTOP,
                                         IdBrowseDictionary);

        labelResults_ = makeControl(hwnd_, L"STATIC", L"Results folder", SS_LEFT, -1);
        results_ = makeControl(hwnd_, L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, IdResults);
        browseResults_ = makeControl(hwnd_, L"BUTTON", L"Browse...", BS_PUSHBUTTON | WS_TABSTOP,
                                     IdBrowseResults);

        labelSeconds_ = makeControl(hwnd_, L"STATIC", L"Seconds (0 = until Stop)", SS_LEFT, -1);
        seconds_ = makeControl(hwnd_, L"EDIT", L"0", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, IdSeconds);
        resident_ = makeControl(hwnd_, L"BUTTON", L"Resident GPU (OpenCL/Metal)",
                                BS_AUTOCHECKBOX | WS_TABSTOP, IdResident);
        unique_ = makeControl(hwnd_, L"BUTTON", L"First address per word", BS_AUTOCHECKBOX | WS_TABSTOP,
                              IdUnique);

        start_ = makeControl(hwnd_, L"BUTTON", L"Start search", BS_DEFPUSHBUTTON | WS_TABSTOP, IdStart);
        stop_ = makeControl(hwnd_, L"BUTTON", L"Stop", BS_PUSHBUTTON | WS_TABSTOP, IdStop);
        selfTest_ = makeControl(hwnd_, L"BUTTON", L"GPU self-test", BS_PUSHBUTTON | WS_TABSTOP, IdSelfTest);
        benchmark_ = makeControl(hwnd_, L"BUTTON", L"Resident benchmark", BS_PUSHBUTTON | WS_TABSTOP,
                                 IdBenchmark);
        openResults_ = makeControl(hwnd_, L"BUTTON", L"Open results", BS_PUSHBUTTON | WS_TABSTOP,
                                   IdOpenResults);
        status_ = makeControl(hwnd_, L"STATIC", L"Ready", SS_LEFT, IdStatus);
        progress_ = makeControl(hwnd_, PROGRESS_CLASSW, L"", PBS_SMOOTH, IdProgress);
        log_ = makeControl(hwnd_, L"EDIT", L"", WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL |
                           ES_READONLY | WS_VSCROLL, IdLog);

        setDefaultPaths();
        for (HWND control : {labelBackend_, backend_, labelDictionary_, dictionary_, browseDictionary_,
                             labelResults_, results_, browseResults_, labelSeconds_, seconds_, resident_,
                             unique_, start_, stop_, selfTest_, benchmark_, openResults_, status_, progress_, log_})
            SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(font_), TRUE);
        EnableWindow(stop_, FALSE);
    }

    void layoutControls(int width, int height) {
        if (!backend_) return;
        const int margin = 12;
        const int labelWidth = 118;
        const int browseWidth = 82;
        const int rowHeight = 27;
        const int editWidth = std::max(220, width - margin * 2 - labelWidth - browseWidth - 12);
        auto place = [](HWND control, int x, int y, int w, int h) {
            MoveWindow(control, x, y, w, h, TRUE);
        };
        place(labelBackend_, margin, 14, labelWidth, rowHeight);
        place(backend_, margin + labelWidth, 10, 160, 160);
        place(labelSeconds_, margin + labelWidth + 180, 14, 150, rowHeight);
        place(seconds_, margin + labelWidth + 330, 10, 75, rowHeight);
        place(resident_, margin + labelWidth + 420, 10, 190, rowHeight);
        place(unique_, margin + labelWidth + 615, 10, 175, rowHeight);

        place(labelDictionary_, margin, 49, labelWidth, rowHeight);
        place(dictionary_, margin + labelWidth, 45, editWidth, rowHeight);
        place(browseDictionary_, width - margin - browseWidth, 45, browseWidth, rowHeight);
        place(labelResults_, margin, 84, labelWidth, rowHeight);
        place(results_, margin + labelWidth, 80, editWidth, rowHeight);
        place(browseResults_, width - margin - browseWidth, 80, browseWidth, rowHeight);

        int buttonX = margin + labelWidth;
        place(start_, buttonX, 118, 120, rowHeight + 3);
        place(stop_, buttonX + 128, 118, 85, rowHeight + 3);
        place(selfTest_, buttonX + 221, 118, 120, rowHeight + 3);
        place(benchmark_, buttonX + 349, 118, 145, rowHeight + 3);
        place(openResults_, buttonX + 502, 118, 120, rowHeight + 3);

        const int logTop = 161;
        const int statusHeight = 24;
        place(log_, margin, logTop, std::max(100, width - margin * 2),
              std::max(80, height - logTop - statusHeight - 16));
        place(status_, margin, height - statusHeight - 10, std::max(100, width - 220), statusHeight);
        place(progress_, width - 195, height - statusHeight - 7, 183, 18);
    }

    void setDefaultPaths() {
        const auto dir = executableDirectory();
        std::filesystem::path words;
        for (const auto& candidate : {dir / L"words.txt", dir / L"words.long.8plus.txt",
                                      std::filesystem::current_path() / L"words.txt"}) {
            if (std::filesystem::exists(candidate)) {
                words = candidate;
                break;
            }
        }
        if (words.empty()) words = dir / L"words.txt";
        setText(dictionary_, words.wstring());
        setText(results_, (dir / L"results").wstring());
    }

    std::filesystem::path executableDirectory() const {
        std::wstring path(32768, L'\0');
        const DWORD count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (count && count < path.size()) return std::filesystem::path(path.substr(0, count)).parent_path();
        return std::filesystem::current_path();
    }

    std::filesystem::path generatorPath() const {
        return executableDirectory() / L"tron_vanity_generator.exe";
    }

    static void setText(HWND control, const std::wstring& text) {
        SetWindowTextW(control, text.c_str());
    }

    static std::wstring textOf(HWND control) {
        const int length = GetWindowTextLengthW(control);
        std::wstring text(static_cast<size_t>(length) + 1, L'\0');
        if (length) {
            GetWindowTextW(control, text.data(), length + 1);
            text.resize(static_cast<size_t>(length));
        } else {
            text.clear();
        }
        return text;
    }

    std::wstring backendName() const {
        const int length = GetWindowTextLengthW(backend_);
        std::wstring value(static_cast<size_t>(length) + 1, L'\0');
        if (length) {
            GetWindowTextW(backend_, value.data(), length + 1);
            value.resize(static_cast<size_t>(length));
        } else {
            value.clear();
        }
        return value;
    }

    bool checked(HWND control) const {
        return SendMessageW(control, BM_GETCHECK, 0, 0) == BST_CHECKED;
    }

    void handleCommand(int id) {
        switch (id) {
        case IdBrowseDictionary:
            browseDictionary();
            break;
        case IdBrowseResults:
            browseResults();
            break;
        case IdOpenResults:
            openResultsFolder();
            break;
        case IdStart:
            startSearch();
            break;
        case IdStop:
            stopProcess();
            break;
        case IdSelfTest:
            startSelfTest();
            break;
        case IdBenchmark:
            startBenchmark();
            break;
        default:
            break;
        }
    }

    void browseDictionary() {
        wchar_t path[32768]{};
        const std::wstring current = textOf(dictionary_);
        if (current.size() < std::size(path)) std::copy(current.begin(), current.end(), path);
        OPENFILENAMEW dialog{sizeof(dialog)};
        dialog.hwndOwner = hwnd_;
        dialog.lpstrFile = path;
        dialog.nMaxFile = static_cast<DWORD>(std::size(path));
        dialog.lpstrFilter = L"Dictionary files (*.txt)\0*.txt\0All files (*.*)\0*.*\0\0";
        dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
        if (GetOpenFileNameW(&dialog)) setText(dictionary_, path);
    }

    void browseResults() {
        BROWSEINFOW dialog{};
        dialog.hwndOwner = hwnd_;
        dialog.lpszTitle = L"Choose results folder";
        dialog.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
        PIDLIST_ABSOLUTE list = SHBrowseForFolderW(&dialog);
        if (!list) return;
        wchar_t path[MAX_PATH]{};
        if (SHGetPathFromIDListW(list, path)) setText(results_, path);
        CoTaskMemFree(list);
    }

    void openResultsFolder() {
        const auto path = std::filesystem::path(textOf(results_));
        try { std::filesystem::create_directories(path); } catch (...) {}
        ShellExecuteW(hwnd_, L"open", path.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }

    bool baseArguments(std::vector<std::wstring>& arguments) {
        const auto generator = generatorPath();
        if (!std::filesystem::exists(generator)) {
            MessageBoxW(hwnd_,
                        L"tron_vanity_generator.exe was not found next to the GUI.\n"
                        L"Copy both files from the release archive into one folder.",
                        L"Generator not found", MB_OK | MB_ICONERROR);
            return false;
        }
        const auto words = std::filesystem::path(textOf(dictionary_));
        const auto results = std::filesystem::path(textOf(results_));
        if (words.empty() || !std::filesystem::exists(words)) {
            MessageBoxW(hwnd_, L"Choose an existing dictionary file first.", L"Dictionary", MB_OK | MB_ICONWARNING);
            return false;
        }
        if (results.empty()) {
            MessageBoxW(hwnd_, L"Choose a results folder first.", L"Results", MB_OK | MB_ICONWARNING);
            return false;
        }
        arguments = {L"--no-config", L"--backend", backendName(), L"--words", words.wstring(),
                     L"--out", results.wstring()};
        return true;
    }

    void startSearch() {
        if (running_) return;
        std::vector<std::wstring> args;
        if (!baseArguments(args)) return;
        std::wstring seconds = textOf(seconds_);
        if (seconds.empty()) seconds = L"0";
        args.insert(args.end(), {L"--seconds", seconds});
        if (checked(resident_) && backendName() != L"vulkan" && backendName() != L"cpu")
            args.push_back(L"--gpu-resident");
        if (checked(unique_)) args.push_back(L"--unique-words");
        startProcess(args, L"Searching");
    }

    void startSelfTest() {
        if (running_) return;
        std::vector<std::wstring> args = {L"--no-config", L"--backend", backendName(), L"--gputest"};
        if (checked(resident_) && backendName() == L"opencl") args.push_back(L"--gpu-resident");
        startProcess(args, L"GPU self-test");
    }

    void startBenchmark() {
        if (running_) return;
        std::vector<std::wstring> args;
        if (!baseArguments(args)) return;
        args.insert(args.end(), {L"--bench-resident", L"--bench-seconds", L"5"});
        startProcess(args, L"Resident benchmark");
    }

    void startProcess(const std::vector<std::wstring>& arguments, const std::wstring& label) {
        if (running_) return;
        const auto generator = generatorPath();
        std::wstring command = quoteArg(generator.wstring());
        for (const auto& argument : arguments) command += L" " + quoteArg(argument);
        std::vector<wchar_t> commandLine(command.begin(), command.end());
        commandLine.push_back(L'\0');

        SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
        HANDLE readPipe = nullptr, writePipe = nullptr;
        if (!CreatePipe(&readPipe, &writePipe, &security, 0)) {
            showLastError(L"Could not create output pipe");
            return;
        }
        if (!SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0)) {
            CloseHandle(readPipe);
            CloseHandle(writePipe);
            showLastError(L"Could not configure output pipe");
            return;
        }

        STARTUPINFOW startup{sizeof(startup)};
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdOutput = writePipe;
        startup.hStdError = writePipe;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        PROCESS_INFORMATION processInfo{};
        const std::wstring cwd = executableDirectory().wstring();
        const BOOL created = CreateProcessW(generator.wstring().c_str(), commandLine.data(), nullptr, nullptr,
                                            TRUE, CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, nullptr,
                                            cwd.c_str(), &startup, &processInfo);
        CloseHandle(writePipe);
        if (!created) {
            CloseHandle(readPipe);
            showLastError(L"Could not start tron_vanity_generator.exe");
            return;
        }
        CloseHandle(processInfo.hThread);
        process_ = processInfo.hProcess;
        running_ = true;
        setText(status_, label + L" running...");
        appendLog(L"\r\n--- " + label + L" ---\r\n");
        EnableWindow(start_, FALSE);
        EnableWindow(selfTest_, FALSE);
        EnableWindow(benchmark_, FALSE);
        EnableWindow(stop_, TRUE);

        reader_ = std::thread([this, readPipe, child = processInfo.hProcess] {
            char buffer[4096];
            DWORD count = 0;
            while (ReadFile(readPipe, buffer, sizeof(buffer), &count, nullptr) && count) {
                auto* output = new std::wstring(redact(fromUtf8(std::string(buffer, buffer + count))));
                if (!PostMessageW(hwnd_, WM_GUI_OUTPUT, 0, reinterpret_cast<LPARAM>(output))) delete output;
            }
            CloseHandle(readPipe);
            WaitForSingleObject(child, INFINITE);
            DWORD exitCode = 1;
            GetExitCodeProcess(child, &exitCode);
            auto* result = new DWORD(exitCode);
            if (!PostMessageW(hwnd_, WM_GUI_EXIT, 0, reinterpret_cast<LPARAM>(result))) delete result;
        });
    }

    void stopProcess() {
        if (!running_ || !process_) return;
        setText(status_, L"Stopping...");
        TerminateProcess(process_, 130);
    }

    void finishProcess(DWORD exitCode) {
        if (reader_.joinable()) reader_.join();
        if (process_) {
            CloseHandle(process_);
            process_ = nullptr;
        }
        running_ = false;
        EnableWindow(start_, TRUE);
        EnableWindow(selfTest_, TRUE);
        EnableWindow(benchmark_, TRUE);
        EnableWindow(stop_, FALSE);
        setText(status_, exitCode == 0 ? L"Finished successfully" :
                (exitCode == 130 ? L"Stopped" : L"Finished with an error"));
        if (exitCode != 0) appendLog(L"\r\nProcess exit code: " + std::to_wstring(exitCode) + L"\r\n");
    }

    void shutdownProcess() {
        if (process_) TerminateProcess(process_, 130);
        if (reader_.joinable()) reader_.join();
        if (process_) {
            CloseHandle(process_);
            process_ = nullptr;
        }
        running_ = false;
    }

    void appendLog(const std::wstring& text) {
        if (!log_) return;
        const std::wstring safe = redact(text);
        const int length = GetWindowTextLengthW(log_);
        if (length > 200000) SetWindowTextW(log_, L"[earlier output trimmed]\r\n");
        SendMessageW(log_, EM_SETSEL, static_cast<WPARAM>(-1), static_cast<LPARAM>(-1));
        SendMessageW(log_, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(safe.c_str()));

        size_t end = safe.find_last_not_of(L"\r\n");
        if (end != std::wstring::npos) {
            size_t begin = safe.find_last_of(L"\r\n", end);
            const std::wstring line = safe.substr(begin == std::wstring::npos ? 0 : begin + 1,
                                                  end - (begin == std::wstring::npos ? 0 : begin + 1) + 1);
            if (line.size() < 220) setText(status_, line);
            const size_t run = line.find(L"| run ");
            if (run != std::wstring::npos) {
                wchar_t* parsedEnd = nullptr;
                const double percent = std::wcstod(line.c_str() + run + 6, &parsedEnd);
                if (parsedEnd != line.c_str() + run + 6 && percent >= 0.0 && percent <= 100.0)
                    SendMessageW(progress_, PBM_SETPOS, static_cast<WPARAM>(percent), 0);
            }
        }
    }

    void showLastError(const wchar_t* prefix) {
        const DWORD error = GetLastError();
        wchar_t message[512]{};
        FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, error,
                      0, message, static_cast<DWORD>(std::size(message)), nullptr);
        std::wstring text = std::wstring(prefix) + L"\n" + message;
        MessageBoxW(hwnd_, text.c_str(), L"TRON Vanity Generator", MB_OK | MB_ICONERROR);
    }

    HINSTANCE instance_ = nullptr;
    HWND hwnd_ = nullptr;
    HWND labelBackend_ = nullptr;
    HWND backend_ = nullptr;
    HWND labelDictionary_ = nullptr;
    HWND dictionary_ = nullptr;
    HWND browseDictionary_ = nullptr;
    HWND labelResults_ = nullptr;
    HWND results_ = nullptr;
    HWND browseResults_ = nullptr;
    HWND labelSeconds_ = nullptr;
    HWND seconds_ = nullptr;
    HWND resident_ = nullptr;
    HWND unique_ = nullptr;
    HWND start_ = nullptr;
    HWND stop_ = nullptr;
    HWND selfTest_ = nullptr;
    HWND benchmark_ = nullptr;
    HWND openResults_ = nullptr;
    HWND status_ = nullptr;
    HWND progress_ = nullptr;
    HWND log_ = nullptr;
    HFONT font_ = nullptr;
    HANDLE process_ = nullptr;
    std::thread reader_;
    std::atomic<bool> running_{false};
};

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    SetProcessDPIAware();
    GuiApp app;
    return app.run(instance);
}
