#include <MsgBoxConstants.au3>

$hwnd = WinActivate("[CLASS:Premiere Pro]", "")
If IsHWnd($hwnd) Then
    If ControlFocus($hwnd, "", "[CLASS:DroverLord - Window Class;INSTANCE:33]") Then
        Send("^a")
        Send("^d")
    Else
        MsgBox($MB_SYSTEMMODAL + $MB_ICONWARNING, "Warning", "Window, but not control activated" & @CRLF & @CRLF & ".")
    EndIf
Else
    ; Notepad will be displayed as MsgBox introduce a delay and allow it.
    MsgBox($MB_SYSTEMMODAL, "", "Window not activated" & @CRLF & @CRLF & ".", 5)
EndIf
