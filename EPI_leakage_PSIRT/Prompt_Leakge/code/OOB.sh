state=0
r1=0
r2=0

START_VAL="0x878d2100158af0"
STOP_VAL="0x878d2000158af0"

while true; do
    RESULT2=$(peci_cmds RdIAMSR 0 0x610 | awk '{print $2}')

    if [[ $state -eq 0 ]]; then
        if [[ "$RESULT2" == "$START_VAL" ]]; then
            state=1
            r1=$(peci_cmds RdPkgConfig 6 0x0000 | awk '{print $2}')
        else
            sleep 0.01
        fi
    else
        sleep 0.06 
        RESULT2=$(peci_cmds RdIAMSR 0 0x610 | awk '{print $2}')

        if [[ "$RESULT2" == "$STOP_VAL" ]]; then
            state=0
        else
            r2=$(peci_cmds RdPkgConfig 6 0x0000 | awk '{print $2}')
            echo $((r2 - r1))
            r1=$r2
        fi
    fi
done
