#!/usr/bin/env bash

# subasta.sh
DESC='Preprocess BIDS neuro data using standard containerized workflows.'

# Author: Isaac David <isacdaavid@at@isacdaavid@dot@info>
# License: GNU General Public License v3 or later.


# Default values for command options.
ATTEMPTS_PER_SUBJECT=1
ATTEMPTS_SLEEP_PERIOD=5
CONTAINER_MEMORY=$(free --gibi --total | tail -n 1 | awk '{print $2}')
CONTAINER_THREADS=$(nproc)
DELETE_ORIGINALS=no
DELETE_ORIGINALS_FREESURFER=no
PARALLEL_SUBJECTS=1

# Mandatory arguments:
WORKFLOW=''
# Directory must be writeable. Either:
# 1) Copy the database to your home (recommended).
# 2) Create an alternative  mountpoint with write permissions
#    (not recommended, filesystem failures are common when using the NAS).
#    To mount SAMBA/CIFS/Windows network volume with write permissions use:
#        gio mount smb://tononi-nas.ad.wisc.edu/white_elephant
#    Then look for the path under /run/user/$UID/...
DIR=''


# Usage function
usage() {
    cat <<EOF
$DESC
Usage: $0 [optional] <workflow> <directory>

<workflow>: fmriprep, qsiprep, xcpd, qsirecon

<directory> must be writable and contain a <directory>/bids/ subdirectory.

Optional:
  --attempts-per-subject <n>        Defaults to ${ATTEMPTS_PER_SUBJECT}.
  --attempts-sleep-period <seconds> Defaults to ${ATTEMPTS_SLEEP_PERIOD}.
  --container-memory <GiB>          Limit memory of each docker instance
                                      (also see --parallel_subjects).
  --container-threads <n>           Limit threads of each docker instance
                                      (also see --parallel_subjects).
  --delete-originals                Remove original subject folder if
                                      processing attempt was successful.
  --delete-originals-freesurfer     Remove freesurfer subject folder if
                                      processing attempt was successful.
  --parallel-subjects <n>           Process many subjects in parallel
                                      docker instances. Defaults to ${PARALLEL_SUBJECTS}.
EOF
}


# Display error message after wrong invocation and exit with error.
invalid_options() {
    if [[ -n "$1" ]]
    then
        echo "Error: invalid option $1." >&2
    else
        echo "Error: missing mandatory arguments." >&2
    fi
    echo "       See $0 --help" >&2
    exit 1
}


# Process options and set globals.
parse_arguments() {
    while [[ $# -gt 2 ]] || [[ "$1" == '--help' ]] || [[ "$1" == '-h' ]]
    do
        case "$1" in
            --attempts-per-subject)
                ATTEMPTS_PER_SUBJECT="$2"
                shift 2
                ;;
            --attempts-sleep-period)
                ATTEMPTS_SLEEP_PERIOD="$2"
                shift 2
                ;;
            --container-memory)
                CONTAINER_MEMORY="$2"
                shift 2
                ;;
            --container-threads)
                CONTAINER_THREADS="$2"
                shift 2
                ;;
            --delete-originals)
                DELETE_ORIGINALS="yes"
                shift
                ;;
            --delete-originals-freesurfer)
                DELETE_ORIGINALS_FREESURFER="yes"
                shift
                ;;
            --parallel-subjects)
                PARALLEL_SUBJECTS="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            *)
                invalid_options "$1"
                ;;
        esac
    done

    # Check for mandatory arguments (WORKFLOW and DIR).
    if [[ $# -lt 2 ]]
    then
        invalid_options
    fi

    WORKFLOW=$1
    DIR=$2

    # Make sure DIR is absolute.
    if [[ "$DIR" != /* ]]
    then
        DIR=${PWD}/${DIR}
    fi
}


# Check if first argument conforms to integer regex.
is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}


# Validate options meant to contain numerical values.
validate_numeric_options() {
    if ! is_positive_integer "$ATTEMPTS_PER_SUBJECT"
    then
        echo 'Error: --attempts-per-subject must be a positive integer.' >&2
        exit 1
    fi

    if ! is_positive_integer "$ATTEMPTS_SLEEP_PERIOD"
    then
        echo 'Error: --attempts-sleep-period must be a positive integer.' >&2
        exit 1
    fi

    if ! is_positive_integer "$CONTAINER_MEMORY"
    then
        echo 'Error: --container-memory must be a positive integer.' >&2
        exit 1
    fi

    if ! is_positive_integer "$CONTAINER_THREADS"
    then
        echo 'Error: --container-threads must be a positive integer.' >&2
        exit 1
    fi

    if ! is_positive_integer "$PARALLEL_SUBJECTS"
    then
        echo 'Error: --parallel-subjects must be a positive integer.' >&2
        exit 1
    fi
}


validate_directory() {
    if [[ ! -d "$DIR" ]]
    then
        echo 'Error: directory not found: %s\n' "$DIR" >&2
        exit 1
    fi

    if [[ ! -w "$DIR" ]]
    then
        printf 'Error: directory is not writable: %s\n' "$DIR" >&2
        exit 1
    fi

    if [[ ! -d "${DIR}/bids" ]]
    then
        printf 'Error: missing BIDS directory: %s/bids\n' "$DIR" >&2
        exit 1
    fi

    if [[ ! -f "${DIR}/bids/dataset_description.json" ]]
    then
        printf 'Error: missing %s/bids/dataset_description.json\n' "$DIR" >&2
        exit 1
    fi

    if [[ ! -f "${DIR}/license.txt" ]]
    then
        printf 'Error: missing freesurfer license %s/license.txt\n' "$DIR" >&2
        exit 1
    fi

    if [[ "$WORKFLOW" == xcpd ]] && [[ ! -f "${DIR}/xcpd.yml" ]]
    then
        printf 'Error: missing nuisance regressors: %s/xcpd.yml\n' "$DIR" >%2
        exit 1
    fi
}


# Check command existence.
require_command() {
    command -v "$1" > /dev/null 2>&1 || {
        printf 'Error: required command not found: %s\n' "$1" >&2
        exit 1
    }
}


# Basic check for external dependencies.
validate_environment() {
    require_command docker
    require_command grep
    require_command awk
    require_command free
    require_command nproc
}


# Master validator.
validate_arguments() {
    validate_numeric_options
    validate_directory
    validate_environment
}


# Ensure docker image exists locally.
# Pull image if absent.
require_docker_image() {
    local image=$1

    if ! docker image inspect "$image" > /dev/null 2>&1
    then
        printf 'Docker image not found locally: %s\n' "$image" >&2
        printf 'Pulling image...\n' >&2

        if ! docker pull "$image"
        then
            printf 'Error: failed to pull docker image: %s\n' "$image" >&2
            exit 1
        fi
    fi
}


# Configure workspace-specific globals.
configure_workflow() {
    case "$WORKFLOW" in
        fmriprep)
            declare -gA FMRIPREP
            FMRIPREP[image]='nipreps/fmriprep:latest'
            FMRIPREP[cifti-output]=91k
            ORIGINALS_BASE="bids"
            ORIGINALS="${DIR}/${ORIGINALS_BASE}"
            DERIVATIVES_BASE="bids/derivatives/${WORKFLOW}"
            DERIVATIVES="${DIR}/${DERIVATIVES_BASE}/"
            require_docker_image "${FMRIPREP[image]}"
            ;;
        qsiprep)
            declare -gA QSIPREP
            QSIPREP[image]='pennlinc/qsiprep:latest'
            QSIPREP[output-resolution]=2
            ORIGINALS_BASE="bids"
            ORIGINALS="${DIR}/${ORIGINALS_BASE}"
            DERIVATIVES_BASE="derivatives/${WORKFLOW}"
            DERIVATIVES="${DIR}/${DERIVATIVES_BASE}/"
            require_docker_image "${QSIPREP[image]}"
            ;;
        xcpd)
            declare -gA XCPD
            XCPD[image]='pennlinc/xcp_d:latest'
            XCPD[file-format]=cifti
            XCPD[despike]=y               # re-interpolate extreme BOLD values.
            XCPD[nuisance-regressors]=xcpd.yml
            XCPD[smoothing]=3             # mm.
            XCPD[combine-runs]=n
            XCPD[motion-filter-type]=none # notch, lp (low-pass).
            XCPD[head-radius]=50          # mm.
            XCPD[fd-thresh]=0.5           # framewise displacement.
            XCPD[min-time]=0              # min data required after censoring.
            XCPD[output-type]=censored    # scrub volumes avobe fd-thresh.
            XCPD[lower-bpf]=0.01
            XCPD[upper-bpf]=0.1
            XCPD[bpf-order]=5
            XCPD[atlases]='4S1056Parcels \
                4S156Parcels \
                4S256Parcels \
                4S356Parcels \
                4S456Parcels \
                4S556Parcels \
                4S656Parcels \
                4S756Parcels \
                4S856Parcels \
                4S956Parcels \
                Glasser \
                Gordon \
                HCP \
                MIDB \
                MyersLabonte \
                Tian'
            ORIGINALS_BASE="bids/derivatives/fmriprep"
            ORIGINALS="${DIR}/${ORIGINALS_BASE}/"
            DERIVATIVES_BASE="bids/derivatives/${WORKFLOW}"
            DERIVATIVES="${DIR}/${DERIVATIVES_BASE}/"
            require_docker_image "${XCPD[image]}"
            ;;
        qsirecon)
            declare -gA QSIRECON
            QSIRECON[image]='pennlinc/qsirecon:latest'
            # TODO: turn into options.
            QSIRECON[recon-spec]=mrtrix_singleshell_ss3t_ACT-hsvs
            QSIRECON[derivatives]=qsirecon-MRtrix3_fork-SS3T_act-HSVS
            # TODO: automatically read resolution from NIFTIs
            QSIRECON[output-resolution]=2 # mm.
            QSIRECON[atlases]='4S156Parcels \
                    4S256Parcels \
                    4S456Parcels \
                    Brainnetome246Ext \
                    AICHA384Ext \
                    Gordon333Ext \
                    AAL116'
            QSIRECON[fs-subjects-dir]="${DIR}/bids/derivatives/fmriprep/sourcedata/freesurfer/"
            ORIGINALS_BASE="bids/derivatives/qsiprep"
            ORIGINALS="${DIR}/${ORIGINALS_BASE}/"
            DERIVATIVES_BASE="bids/derivatives/${WORKFLOW}"
            DERIVATIVES="${DIR}/${DERIVATIVES_BASE}/"
            require_docker_image "${QSIRECON[image]}"
            ;;
        *)
            echo "Error: unknown workflow $WORKFLOW." >&2
            echo "       See $0 --help" >&2
            exit 1
            ;;
    esac
}


# Multiple-dispatch wrapper.
run_workflow() {
    case "$WORKFLOW" in
        fmriprep)
            run_fmriprep "$@" ;;
        xcpd)
            run_xcpd "$@" ;;
        qsiprep)
            run_qsiprep "$@" ;;
        qsirecon)
            run_qsirecon "$@" ;;
    esac
}


# Log processing attempt in stdout.
log_attempt() {
    local sub=$1
    local attempt=$2
    printf "\n\nProcessing subject %s, attempt %s/%s\n\n" \
           "$sub" "$attempt" "$ATTEMPTS_PER_SUBJECT"
}


# Run fmriprep for some subject.
run_fmriprep() {
    local sub=$1
    local attempts=$2
    local attempt=$((ATTEMPTS_PER_SUBJECT - $attempts + 1))
    log_attempt "$sub" "$attempt"
    docker run --cpus="$CONTAINER_THREADS" --memory="$CONTAINER_MEMORY"g \
           -u $(id -u):$(id -g) --rm -v "$DIR":/tmp \
           --entrypoint=fmriprep "${FMRIPREP[image]}" \
               --participant-label "$sub" \
               --fs-license-file license.txt \
               --cifti-output "${FMRIPREP[cifti-output]}" \
              "$ORIGINALS_BASE" \
              "$DERIVATIVES_BASE" \
              participant
    sleep "$ATTEMPTS_SLEEP_PERIOD"
}


# Run qsiprep for some subject.
run_qsiprep() {
    local sub=$1
    local attempts=$2
    local attempt=$((ATTEMPTS_PER_SUBJECT - $attempts + 1))
    log_attempt "$sub" "$attempt"
    docker run --cpus="$CONTAINER_THREADS" --memory="$CONTAINER_MEMORY"g \
           -u $(id -u):$(id -g) --rm -v "$DIR":/tmp \
           --entrypoint=qsiprep "${QSIPREP[image]}" \
               --participant-label "$sub" \
               --fs-license-file license.txt \
               --output-resolution "${QSIPREP[output-resolution]}" \
               "$ORIGINALS_BASE" \
               "$DERIVATIVES_BASE" \
               participant
    sleep "$ATTEMPTS_SLEEP_PERIOD"
}


# Run XCP-D for some subject
run_xcpd() {
    local sub=$1
    local attempts=$2
    local attempt=$((ATTEMPTS_PER_SUBJECT - $attempts + 1))
    log_attempt "$sub" "$attempt"
    # Don't quote atlases, they must arrive as separate arguments.
    docker run --cpus="$CONTAINER_THREADS" --memory="$CONTAINER_MEMORY"g \
           -u $(id -u):$(id -g) --rm -v "$DIR":/tmp \
           --entrypoint=xcp_d "${XCPD[image]}" \
               --mode linc \
               --participant-label "$sub" \
               --input-type fmriprep \
               --file-format ${XCPD[file-format]} \
               --despike ${XCPD[despike]} \
               --nuisance-regressors ${XCPD[nuisance-regressors]} \
               --smoothing ${XCPD[smoothing]} \
               --combine-runs ${XCPD[combine-runs]} \
               --motion-filter-type ${XCPD[motion-filter-type]} \
               --head-radius ${XCPD[head-radius]} \
               --fd-thresh ${XCPD[fd-thresh]} \
               --min-time ${XCPD[min-time]} \
               --output-type ${XCPD[output-type]} \
               --lower-bpf ${XCPD[lower-bpf]} \
               --upper-bpf ${XCPD[upper-bpf]} \
               --bpf-order ${XCPD[bpf-order]} \
               --atlases ${XCPD[atlases]} \
               --fs-license-file license.txt \
               "$ORIGINALS_BASE" \
               "$DERIVATIVES_BASE" \
               participant
    sleep "$ATTEMPTS_SLEEP_PERIOD"
}


# Run qsirecon for some subject
run_qsirecon() {
    local sub=$1
    local attempts=$2
    local attempt=$((ATTEMPTS_PER_SUBJECT - $attempts + 1))
    log_attempt "$sub" "$attempt"
    mv -f -- "${DERIVATIVES}/${sub}"/ses*/anat \
             "${DERIVATIVES}/${sub}"/anat.old > /dev/null
    cp -rf -- "${DERIVATIVES}/${sub}"/"$sub"/anat \
             "${DERIVATIVES}/${sub}"/ses*/ > /dev/null
    # NOTE: qsirecon's argument parsing is buggy, changing option
    # order might break things.
    # Don't quote atlases, they must arrive as separate arguments.
    docker run --cpus="$CONTAINER_THREADS" --memory="$CONTAINER_MEMORY"g \
           -u $(id -u):$(id -g) \
           --rm \
           -v "$DIR":/tmp \
           -v "${QSIRECON[fs-subjects-dir]}":/tmp2 \
           --entrypoint=qsirecon "${QSIRECON[image]}" \
               --participant-label "$sub" \
               --fs-license-file license.txt \
               --recon-spec "${QSIRECON[recon-spec]}" \
               --input-type qsiprep \
               --fs-subjects-dir /tmp2 \
               "$ORIGINALS_BASE" \
               "$DERIVATIVES_BASE" \
               participant \
               --atlases ${QSIRECON[atlases]}
    sleep "$ATTEMPTS_SLEEP_PERIOD"
}


# Remove .html and directory under $DERIVATIVES for some subject.
delete_derivative() {
    local sub=$1
    rm -rf -- "${DERIVATIVES}/${sub}"{,.html} > /dev/null 2>&1
}


# Remove original directory for some subject
# (lower disk usage when working with copy outside SAMBA volume).
delete_original() {
    local sub=$1
    rm -rf -- "${ORIGINALS}/${sub}" > /dev/null 2>&1
}


# Remove original freesurfer outputs for some subject
# (lower disk usage when working with copy outside SAMBA volume).
delete_original_freesurfer() {
    local sub=$1
    local fs_dir="${DIR}/bids/derivatives/fmriprep/sourcedata/freesurfer/${sub}"
    rm -rf -- "$fs_dir" > /dev/null 2>&1
}


# Check whether workflow run was successful for some subject.
is_successful() {
    local sub=$1
    grep -q 'No errors to report' "${DERIVATIVES}/${sub}.html"
    return $?
}


process_subject() {
    local sub=$1
    local attempts=${ATTEMPTS_PER_SUBJECT}

    run_workflow "$sub" "$attempts"
    ((attempts--))

    while true
    do
        if is_successful "$sub"
        then
            if [[ "$DELETE_ORIGINALS" == yes ]]
            then
                delete_original "$sub"
            fi
            if [[ "$DELETE_ORIGINALS_FREESURFER" == yes ]]
            then
                delete_original_freesurfer "$sub"
            fi
            break
        fi

        delete_derivative "$sub"

        if [[ $attempts -le 0 ]]
        then
            break
        fi

        run_workflow "$sub" "$attempts"
        ((attempts--))
    done
}


main() {
    if ! [[ -d "$ORIGINALS" ]]
    then
        echo "Error: directory not found $ORIGINALS"
        exit 1
    fi

    local -a subjects
    mapfile -t subjects < <(
        find "$ORIGINALS" -maxdepth 1 -type d -name 'sub-*'
    )

    if [[ ${#subjects[@]} -eq 0 ]]
    then
        echo 'Error: no subjects found.'
        exit 1
    fi

    for sub in "${subjects[@]}"
    do
        # Remove basedir, we only want subject ID proper.
        local sub=${sub##*/}

        # If subject directory is already found under $DERIVATIVES, skip it.
        if [[ -d "${DERIVATIVES}/${sub}" ]]
        then
            printf "\n\nSkipping subject %s, folder already exists in %s\n\n" \
                   "$sub" "$DERIVATIVES"
            continue
        fi

        # Otherwise, run workflow til successful or til attempts are depleted.
        process_subject "$sub" &

        # Limit number of subjects processed in parallel in the background.
        if [[ $(jobs -r -p | wc -l) -ge $PARALLEL_SUBJECTS ]]
        then
            wait -n
        fi
    done

    # Wait for all background processes to finish
    wait
}


parse_arguments "$@"
validate_arguments
configure_workflow

main
