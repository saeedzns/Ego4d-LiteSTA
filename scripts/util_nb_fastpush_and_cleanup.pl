#!/usr/bin/env perl
use strict; use warnings;
use JSON::PP;

sub read_nb {
  my ($p) = @_;
  open my $fh, '<:raw', $p or die $!;
  local $/; my $txt = <$fh>; close $fh;
  return decode_json($txt);
}

sub write_nb {
  my ($p, $nb) = @_;
  open my $out, '>:raw', $p or die $!;
  binmode($out, ':utf8');
  print $out JSON::PP->new->pretty->encode($nb);
  close $out;
}

sub src_text {
  my ($cell) = @_;
  my $a = $cell->{source};
  return '' unless $a && ref($a) eq 'ARRAY';
  return join('', @$a);
}

sub ensure_env_before_fast_push {
  my ($nb) = @_;
  my $cells = $nb->{cells} || [];
  my $inserted = 0;
  for (my $i = 0; $i < scalar(@$cells); $i++) {
    my $c = $cells->[$i];
    my $src = src_text($c);
    next unless $src =~ /!FAST_PUSH=1\s+bash\s+push\.sh/;
    my $need = 1;
    if ($i >= 1) {
      my $prev = $cells->[$i-1];
      my $psrc = src_text($prev);
      if (($prev->{cell_type}||'') eq 'code' && $psrc =~ /%env\s+FAST_PUSH_INCLUDE_HEAVY=1/) {
        $need = 0;
      }
    }
    if ($need) {
      my $env_cell = {
        cell_type => 'code',
        metadata  => { egolite_sync => 'fast_push', id => 'envheavy-'.int(rand(1e9)) },
        execution_count => JSON::PP::null(),
        outputs => [],
        source => [ "%env FAST_PUSH_INCLUDE_HEAVY=1\n" ],
      };
      splice(@$cells, $i, 0, $env_cell);
      $i++;
      $inserted++;
    }
  }
  $nb->{cells} = $cells if $inserted;
  return $inserted;
}

sub add_cleanup_section {
  my ($nb) = @_;
  for my $c (@{$nb->{cells} || []}) {
    my $src = src_text($c);
    return 0 if $src =~ /History Cleanup: remove awscliv2\.zip/;
  }
  my $md = {
    cell_type => 'markdown',
    metadata  => { egolite_sync => 'cleanup', id => 'cleanup-md-'.int(rand(1e9)) },
    source => [
      "## History Cleanup: remove awscliv2.zip (optional)\n",
      "This rewrites repo history to drop the large awscliv2.zip blobs.\n",
      "- Runs on /content for speed.\n",
      "- Force-push affects all collaborators; ensure you are the only active pusher.\n",
      "- After success, consider recloning locally to avoid dangling objects.\n",
    ],
  };
  my $code = {
    cell_type => 'code',
    metadata  => { egolite_sync => 'cleanup', id => 'cleanup-code-'.int(rand(1e9)) },
    execution_count => JSON::PP::null(),
    outputs => [],
    source => [
      '%%bash -s "$WORKDIR"\n',
      'set -euo pipefail\n',
      'WORKDIR="$1"\n',
      'if [ -z "$WORKDIR" ]; then echo "Set WORKDIR earlier in the notebook." >&2; exit 1; fi\n',
      'origin=$(git -C "$WORKDIR" config --get remote.origin.url)\n',
      'branch=$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD)\n',
      'mirror=/content/egolite_cleanup.git\n',
      'python3 -m pip -q install git-filter-repo\n',
      'rm -rf "$mirror"\n',
      'git clone --mirror "$origin" "$mirror"\n',
      'cd "$mirror"\n',
      'git filter-repo --invert-paths --path awscliv2.zip --force\n',
      'git count-objects -vH\n',
      'echo Origin: $origin\n',
      'echo Branch: $branch\n',
      "echo 'Review above size. If okay, remove # below to force-push:'\n",
      '# git push --force --prune\n',
    ],
  };
  push @{$nb->{cells}}, $md, $code;
  return 1;
}

sub process_notebook {
  my ($path) = @_;
  my $nb = read_nb($path);
  my $a = ensure_env_before_fast_push($nb);
  my $b = add_cleanup_section($nb);
  if ($a || $b) { write_nb($path, $nb); }
  return ($a, $b);
}

my @paths = glob('notebooks/*.ipynb');
my ($total_env, $total_cleanup) = (0,0);
for my $p (@paths) {
  my ($a,$b) = process_notebook($p);
  $total_env += $a; $total_cleanup += $b;
  print "$p: env_cells_inserted=$a, cleanup_added=$b\n";
}
print "Summary: env_cells_inserted=$total_env, cleanup_added=$total_cleanup\n";
